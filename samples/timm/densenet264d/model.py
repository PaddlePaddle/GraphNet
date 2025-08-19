import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_features_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        L_x_: torch.Tensor,
        L_self_modules_features_modules_norm0_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_norm0_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_norm0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_norm0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_transition1_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_transition1_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_transition1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_transition1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_transition1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_transition2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_transition2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_transition2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_transition2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_transition2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_transition3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_transition3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_transition3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_transition3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_transition3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_norm5_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_norm5_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_norm5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_norm5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_features_modules_conv0_parameters_weight_ = (
            L_self_modules_features_modules_conv0_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_features_modules_norm0_buffers_running_mean_ = (
            L_self_modules_features_modules_norm0_buffers_running_mean_
        )
        l_self_modules_features_modules_norm0_buffers_running_var_ = (
            L_self_modules_features_modules_norm0_buffers_running_var_
        )
        l_self_modules_features_modules_norm0_parameters_weight_ = (
            L_self_modules_features_modules_norm0_parameters_weight_
        )
        l_self_modules_features_modules_norm0_parameters_bias_ = (
            L_self_modules_features_modules_norm0_parameters_bias_
        )
        l_self_modules_features_modules_conv1_parameters_weight_ = (
            L_self_modules_features_modules_conv1_parameters_weight_
        )
        l_self_modules_features_modules_norm1_buffers_running_mean_ = (
            L_self_modules_features_modules_norm1_buffers_running_mean_
        )
        l_self_modules_features_modules_norm1_buffers_running_var_ = (
            L_self_modules_features_modules_norm1_buffers_running_var_
        )
        l_self_modules_features_modules_norm1_parameters_weight_ = (
            L_self_modules_features_modules_norm1_parameters_weight_
        )
        l_self_modules_features_modules_norm1_parameters_bias_ = (
            L_self_modules_features_modules_norm1_parameters_bias_
        )
        l_self_modules_features_modules_conv2_parameters_weight_ = (
            L_self_modules_features_modules_conv2_parameters_weight_
        )
        l_self_modules_features_modules_norm2_buffers_running_mean_ = (
            L_self_modules_features_modules_norm2_buffers_running_mean_
        )
        l_self_modules_features_modules_norm2_buffers_running_var_ = (
            L_self_modules_features_modules_norm2_buffers_running_var_
        )
        l_self_modules_features_modules_norm2_parameters_weight_ = (
            L_self_modules_features_modules_norm2_parameters_weight_
        )
        l_self_modules_features_modules_norm2_parameters_bias_ = (
            L_self_modules_features_modules_norm2_parameters_bias_
        )
        l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer1_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer2_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer3_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer4_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer5_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock1_modules_denselayer6_modules_conv2_parameters_weight_
        l_self_modules_features_modules_transition1_modules_norm_buffers_running_mean_ = L_self_modules_features_modules_transition1_modules_norm_buffers_running_mean_
        l_self_modules_features_modules_transition1_modules_norm_buffers_running_var_ = L_self_modules_features_modules_transition1_modules_norm_buffers_running_var_
        l_self_modules_features_modules_transition1_modules_norm_parameters_weight_ = (
            L_self_modules_features_modules_transition1_modules_norm_parameters_weight_
        )
        l_self_modules_features_modules_transition1_modules_norm_parameters_bias_ = (
            L_self_modules_features_modules_transition1_modules_norm_parameters_bias_
        )
        l_self_modules_features_modules_transition1_modules_conv_parameters_weight_ = (
            L_self_modules_features_modules_transition1_modules_conv_parameters_weight_
        )
        l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer1_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer2_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer3_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer4_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer5_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer6_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer7_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer8_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer9_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer10_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer11_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock2_modules_denselayer12_modules_conv2_parameters_weight_
        l_self_modules_features_modules_transition2_modules_norm_buffers_running_mean_ = L_self_modules_features_modules_transition2_modules_norm_buffers_running_mean_
        l_self_modules_features_modules_transition2_modules_norm_buffers_running_var_ = L_self_modules_features_modules_transition2_modules_norm_buffers_running_var_
        l_self_modules_features_modules_transition2_modules_norm_parameters_weight_ = (
            L_self_modules_features_modules_transition2_modules_norm_parameters_weight_
        )
        l_self_modules_features_modules_transition2_modules_norm_parameters_bias_ = (
            L_self_modules_features_modules_transition2_modules_norm_parameters_bias_
        )
        l_self_modules_features_modules_transition2_modules_conv_parameters_weight_ = (
            L_self_modules_features_modules_transition2_modules_conv_parameters_weight_
        )
        l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer1_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer2_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer3_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer4_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer5_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer6_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer7_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer8_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer9_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer10_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer11_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer12_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer13_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer14_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer15_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer16_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer17_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer18_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer19_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer20_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer21_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer22_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer23_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer24_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer25_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer26_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer27_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer28_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer29_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer30_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer31_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer32_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer33_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer34_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer35_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer36_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer37_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer38_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer39_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer40_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer41_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer42_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer43_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer44_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer45_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer46_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer47_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer48_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer49_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer50_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer51_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer52_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer53_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer54_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer55_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer56_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer57_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer58_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer59_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer60_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer61_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer62_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer63_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock3_modules_denselayer64_modules_conv2_parameters_weight_
        l_self_modules_features_modules_transition3_modules_norm_buffers_running_mean_ = L_self_modules_features_modules_transition3_modules_norm_buffers_running_mean_
        l_self_modules_features_modules_transition3_modules_norm_buffers_running_var_ = L_self_modules_features_modules_transition3_modules_norm_buffers_running_var_
        l_self_modules_features_modules_transition3_modules_norm_parameters_weight_ = (
            L_self_modules_features_modules_transition3_modules_norm_parameters_weight_
        )
        l_self_modules_features_modules_transition3_modules_norm_parameters_bias_ = (
            L_self_modules_features_modules_transition3_modules_norm_parameters_bias_
        )
        l_self_modules_features_modules_transition3_modules_conv_parameters_weight_ = (
            L_self_modules_features_modules_transition3_modules_conv_parameters_weight_
        )
        l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer1_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer2_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer3_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer4_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer5_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer6_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer7_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer8_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer9_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer10_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer11_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer12_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer13_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer14_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer15_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer16_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer17_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer18_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer19_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer20_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer21_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer22_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer23_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer24_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer25_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer26_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer27_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer28_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer29_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer30_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer31_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer32_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer33_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer34_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer35_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer36_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer37_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer38_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer39_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer40_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer41_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer42_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer43_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer44_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer45_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer46_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer47_modules_conv2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_conv1_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_conv1_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_buffers_running_mean_ = L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_buffers_running_mean_
        l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_buffers_running_var_ = L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_buffers_running_var_
        l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_parameters_weight_
        l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_parameters_bias_ = L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_parameters_bias_
        l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_conv2_parameters_weight_ = L_self_modules_features_modules_denseblock4_modules_denselayer48_modules_conv2_parameters_weight_
        l_self_modules_features_modules_norm5_buffers_running_mean_ = (
            L_self_modules_features_modules_norm5_buffers_running_mean_
        )
        l_self_modules_features_modules_norm5_buffers_running_var_ = (
            L_self_modules_features_modules_norm5_buffers_running_var_
        )
        l_self_modules_features_modules_norm5_parameters_weight_ = (
            L_self_modules_features_modules_norm5_parameters_weight_
        )
        l_self_modules_features_modules_norm5_parameters_bias_ = (
            L_self_modules_features_modules_norm5_parameters_bias_
        )
        l_self_modules_classifier_parameters_weight_ = (
            L_self_modules_classifier_parameters_weight_
        )
        l_self_modules_classifier_parameters_bias_ = (
            L_self_modules_classifier_parameters_bias_
        )
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_features_modules_conv0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_features_modules_conv0_parameters_weight_ = None
        x = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_features_modules_norm0_buffers_running_mean_,
            l_self_modules_features_modules_norm0_buffers_running_var_,
            l_self_modules_features_modules_norm0_parameters_weight_,
            l_self_modules_features_modules_norm0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = (
            l_self_modules_features_modules_norm0_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_norm0_buffers_running_var_
        ) = (
            l_self_modules_features_modules_norm0_parameters_weight_
        ) = l_self_modules_features_modules_norm0_parameters_bias_ = None
        x_1 = torch.nn.functional.relu(x, inplace=True)
        x = None
        input_2 = torch.conv2d(
            x_1,
            l_self_modules_features_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_1 = l_self_modules_features_modules_conv1_parameters_weight_ = None
        x_2 = torch.nn.functional.batch_norm(
            input_2,
            l_self_modules_features_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_2 = (
            l_self_modules_features_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_features_modules_norm1_parameters_weight_
        ) = l_self_modules_features_modules_norm1_parameters_bias_ = None
        x_3 = torch.nn.functional.relu(x_2, inplace=True)
        x_2 = None
        input_3 = torch.conv2d(
            x_3,
            l_self_modules_features_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_3 = l_self_modules_features_modules_conv2_parameters_weight_ = None
        x_4 = torch.nn.functional.batch_norm(
            input_3,
            l_self_modules_features_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_3 = (
            l_self_modules_features_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_features_modules_norm2_parameters_weight_
        ) = l_self_modules_features_modules_norm2_parameters_bias_ = None
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        input_4 = torch.nn.functional.max_pool2d(
            x_5, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_5 = None
        concated_features = torch.cat([input_4], 1)
        x_6 = torch.nn.functional.batch_norm(
            concated_features,
            l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features = l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm1_parameters_bias_ = (None)
        x_7 = torch.nn.functional.relu(x_6, inplace=True)
        x_6 = None
        bottleneck_output = torch.conv2d(
            x_7,
            l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_7 = l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_conv1_parameters_weight_ = (None)
        x_8 = torch.nn.functional.batch_norm(
            bottleneck_output,
            l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output = l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_norm2_parameters_bias_ = (None)
        x_9 = torch.nn.functional.relu(x_8, inplace=True)
        x_8 = None
        new_features = torch.conv2d(
            x_9,
            l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_conv2_parameters_weight_ = (None)
        concated_features_1 = torch.cat([input_4, new_features], 1)
        x_10 = torch.nn.functional.batch_norm(
            concated_features_1,
            l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_1 = l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm1_parameters_bias_ = (None)
        x_11 = torch.nn.functional.relu(x_10, inplace=True)
        x_10 = None
        bottleneck_output_1 = torch.conv2d(
            x_11,
            l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_11 = l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_conv1_parameters_weight_ = (None)
        x_12 = torch.nn.functional.batch_norm(
            bottleneck_output_1,
            l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_1 = l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_norm2_parameters_bias_ = (None)
        x_13 = torch.nn.functional.relu(x_12, inplace=True)
        x_12 = None
        new_features_1 = torch.conv2d(
            x_13,
            l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_conv2_parameters_weight_ = (None)
        concated_features_2 = torch.cat([input_4, new_features, new_features_1], 1)
        x_14 = torch.nn.functional.batch_norm(
            concated_features_2,
            l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_2 = l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm1_parameters_bias_ = (None)
        x_15 = torch.nn.functional.relu(x_14, inplace=True)
        x_14 = None
        bottleneck_output_2 = torch.conv2d(
            x_15,
            l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_15 = l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_conv1_parameters_weight_ = (None)
        x_16 = torch.nn.functional.batch_norm(
            bottleneck_output_2,
            l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_2 = l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_norm2_parameters_bias_ = (None)
        x_17 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        new_features_2 = torch.conv2d(
            x_17,
            l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_17 = l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_conv2_parameters_weight_ = (None)
        concated_features_3 = torch.cat(
            [input_4, new_features, new_features_1, new_features_2], 1
        )
        x_18 = torch.nn.functional.batch_norm(
            concated_features_3,
            l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_3 = l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm1_parameters_bias_ = (None)
        x_19 = torch.nn.functional.relu(x_18, inplace=True)
        x_18 = None
        bottleneck_output_3 = torch.conv2d(
            x_19,
            l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_19 = l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_conv1_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            bottleneck_output_3,
            l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_3 = l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_norm2_parameters_bias_ = (None)
        x_21 = torch.nn.functional.relu(x_20, inplace=True)
        x_20 = None
        new_features_3 = torch.conv2d(
            x_21,
            l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_21 = l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_conv2_parameters_weight_ = (None)
        concated_features_4 = torch.cat(
            [input_4, new_features, new_features_1, new_features_2, new_features_3], 1
        )
        x_22 = torch.nn.functional.batch_norm(
            concated_features_4,
            l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_4 = l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm1_parameters_bias_ = (None)
        x_23 = torch.nn.functional.relu(x_22, inplace=True)
        x_22 = None
        bottleneck_output_4 = torch.conv2d(
            x_23,
            l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_23 = l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_conv1_parameters_weight_ = (None)
        x_24 = torch.nn.functional.batch_norm(
            bottleneck_output_4,
            l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_4 = l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_norm2_parameters_bias_ = (None)
        x_25 = torch.nn.functional.relu(x_24, inplace=True)
        x_24 = None
        new_features_4 = torch.conv2d(
            x_25,
            l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_conv2_parameters_weight_ = (None)
        concated_features_5 = torch.cat(
            [
                input_4,
                new_features,
                new_features_1,
                new_features_2,
                new_features_3,
                new_features_4,
            ],
            1,
        )
        x_26 = torch.nn.functional.batch_norm(
            concated_features_5,
            l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_5 = l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm1_parameters_bias_ = (None)
        x_27 = torch.nn.functional.relu(x_26, inplace=True)
        x_26 = None
        bottleneck_output_5 = torch.conv2d(
            x_27,
            l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_27 = l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_conv1_parameters_weight_ = (None)
        x_28 = torch.nn.functional.batch_norm(
            bottleneck_output_5,
            l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_5 = l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_norm2_parameters_bias_ = (None)
        x_29 = torch.nn.functional.relu(x_28, inplace=True)
        x_28 = None
        new_features_5 = torch.conv2d(
            x_29,
            l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_29 = l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_conv2_parameters_weight_ = (None)
        input_5 = torch.cat(
            [
                input_4,
                new_features,
                new_features_1,
                new_features_2,
                new_features_3,
                new_features_4,
                new_features_5,
            ],
            1,
        )
        input_4 = (
            new_features
        ) = (
            new_features_1
        ) = new_features_2 = new_features_3 = new_features_4 = new_features_5 = None
        x_30 = torch.nn.functional.batch_norm(
            input_5,
            l_self_modules_features_modules_transition1_modules_norm_buffers_running_mean_,
            l_self_modules_features_modules_transition1_modules_norm_buffers_running_var_,
            l_self_modules_features_modules_transition1_modules_norm_parameters_weight_,
            l_self_modules_features_modules_transition1_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_5 = l_self_modules_features_modules_transition1_modules_norm_buffers_running_mean_ = l_self_modules_features_modules_transition1_modules_norm_buffers_running_var_ = (
            l_self_modules_features_modules_transition1_modules_norm_parameters_weight_
        ) = (
            l_self_modules_features_modules_transition1_modules_norm_parameters_bias_
        ) = None
        x_31 = torch.nn.functional.relu(x_30, inplace=True)
        x_30 = None
        input_6 = torch.conv2d(
            x_31,
            l_self_modules_features_modules_transition1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_31 = (
            l_self_modules_features_modules_transition1_modules_conv_parameters_weight_
        ) = None
        input_7 = torch._C._nn.avg_pool2d(input_6, 2, 2, 0, False, True, None)
        input_6 = None
        concated_features_6 = torch.cat([input_7], 1)
        x_32 = torch.nn.functional.batch_norm(
            concated_features_6,
            l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_6 = l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm1_parameters_bias_ = (None)
        x_33 = torch.nn.functional.relu(x_32, inplace=True)
        x_32 = None
        bottleneck_output_6 = torch.conv2d(
            x_33,
            l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_33 = l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_conv1_parameters_weight_ = (None)
        x_34 = torch.nn.functional.batch_norm(
            bottleneck_output_6,
            l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_6 = l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_norm2_parameters_bias_ = (None)
        x_35 = torch.nn.functional.relu(x_34, inplace=True)
        x_34 = None
        new_features_6 = torch.conv2d(
            x_35,
            l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_conv2_parameters_weight_ = (None)
        concated_features_7 = torch.cat([input_7, new_features_6], 1)
        x_36 = torch.nn.functional.batch_norm(
            concated_features_7,
            l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_7 = l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm1_parameters_bias_ = (None)
        x_37 = torch.nn.functional.relu(x_36, inplace=True)
        x_36 = None
        bottleneck_output_7 = torch.conv2d(
            x_37,
            l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_conv1_parameters_weight_ = (None)
        x_38 = torch.nn.functional.batch_norm(
            bottleneck_output_7,
            l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_7 = l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_norm2_parameters_bias_ = (None)
        x_39 = torch.nn.functional.relu(x_38, inplace=True)
        x_38 = None
        new_features_7 = torch.conv2d(
            x_39,
            l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_conv2_parameters_weight_ = (None)
        concated_features_8 = torch.cat([input_7, new_features_6, new_features_7], 1)
        x_40 = torch.nn.functional.batch_norm(
            concated_features_8,
            l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_8 = l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm1_parameters_bias_ = (None)
        x_41 = torch.nn.functional.relu(x_40, inplace=True)
        x_40 = None
        bottleneck_output_8 = torch.conv2d(
            x_41,
            l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_41 = l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_conv1_parameters_weight_ = (None)
        x_42 = torch.nn.functional.batch_norm(
            bottleneck_output_8,
            l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_8 = l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_norm2_parameters_bias_ = (None)
        x_43 = torch.nn.functional.relu(x_42, inplace=True)
        x_42 = None
        new_features_8 = torch.conv2d(
            x_43,
            l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_43 = l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_conv2_parameters_weight_ = (None)
        concated_features_9 = torch.cat(
            [input_7, new_features_6, new_features_7, new_features_8], 1
        )
        x_44 = torch.nn.functional.batch_norm(
            concated_features_9,
            l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_9 = l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm1_parameters_bias_ = (None)
        x_45 = torch.nn.functional.relu(x_44, inplace=True)
        x_44 = None
        bottleneck_output_9 = torch.conv2d(
            x_45,
            l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_45 = l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_conv1_parameters_weight_ = (None)
        x_46 = torch.nn.functional.batch_norm(
            bottleneck_output_9,
            l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_9 = l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_norm2_parameters_bias_ = (None)
        x_47 = torch.nn.functional.relu(x_46, inplace=True)
        x_46 = None
        new_features_9 = torch.conv2d(
            x_47,
            l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_47 = l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_conv2_parameters_weight_ = (None)
        concated_features_10 = torch.cat(
            [input_7, new_features_6, new_features_7, new_features_8, new_features_9], 1
        )
        x_48 = torch.nn.functional.batch_norm(
            concated_features_10,
            l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_10 = l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm1_parameters_bias_ = (None)
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        bottleneck_output_10 = torch.conv2d(
            x_49,
            l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_conv1_parameters_weight_ = (None)
        x_50 = torch.nn.functional.batch_norm(
            bottleneck_output_10,
            l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_10 = l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_norm2_parameters_bias_ = (None)
        x_51 = torch.nn.functional.relu(x_50, inplace=True)
        x_50 = None
        new_features_10 = torch.conv2d(
            x_51,
            l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_51 = l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_conv2_parameters_weight_ = (None)
        concated_features_11 = torch.cat(
            [
                input_7,
                new_features_6,
                new_features_7,
                new_features_8,
                new_features_9,
                new_features_10,
            ],
            1,
        )
        x_52 = torch.nn.functional.batch_norm(
            concated_features_11,
            l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_11 = l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm1_parameters_bias_ = (None)
        x_53 = torch.nn.functional.relu(x_52, inplace=True)
        x_52 = None
        bottleneck_output_11 = torch.conv2d(
            x_53,
            l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_conv1_parameters_weight_ = (None)
        x_54 = torch.nn.functional.batch_norm(
            bottleneck_output_11,
            l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_11 = l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_norm2_parameters_bias_ = (None)
        x_55 = torch.nn.functional.relu(x_54, inplace=True)
        x_54 = None
        new_features_11 = torch.conv2d(
            x_55,
            l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_55 = l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_conv2_parameters_weight_ = (None)
        concated_features_12 = torch.cat(
            [
                input_7,
                new_features_6,
                new_features_7,
                new_features_8,
                new_features_9,
                new_features_10,
                new_features_11,
            ],
            1,
        )
        x_56 = torch.nn.functional.batch_norm(
            concated_features_12,
            l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_12 = l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm1_parameters_bias_ = (None)
        x_57 = torch.nn.functional.relu(x_56, inplace=True)
        x_56 = None
        bottleneck_output_12 = torch.conv2d(
            x_57,
            l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_conv1_parameters_weight_ = (None)
        x_58 = torch.nn.functional.batch_norm(
            bottleneck_output_12,
            l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_12 = l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_norm2_parameters_bias_ = (None)
        x_59 = torch.nn.functional.relu(x_58, inplace=True)
        x_58 = None
        new_features_12 = torch.conv2d(
            x_59,
            l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_59 = l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_conv2_parameters_weight_ = (None)
        concated_features_13 = torch.cat(
            [
                input_7,
                new_features_6,
                new_features_7,
                new_features_8,
                new_features_9,
                new_features_10,
                new_features_11,
                new_features_12,
            ],
            1,
        )
        x_60 = torch.nn.functional.batch_norm(
            concated_features_13,
            l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_13 = l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm1_parameters_bias_ = (None)
        x_61 = torch.nn.functional.relu(x_60, inplace=True)
        x_60 = None
        bottleneck_output_13 = torch.conv2d(
            x_61,
            l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_61 = l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_conv1_parameters_weight_ = (None)
        x_62 = torch.nn.functional.batch_norm(
            bottleneck_output_13,
            l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_13 = l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_norm2_parameters_bias_ = (None)
        x_63 = torch.nn.functional.relu(x_62, inplace=True)
        x_62 = None
        new_features_13 = torch.conv2d(
            x_63,
            l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_conv2_parameters_weight_ = (None)
        concated_features_14 = torch.cat(
            [
                input_7,
                new_features_6,
                new_features_7,
                new_features_8,
                new_features_9,
                new_features_10,
                new_features_11,
                new_features_12,
                new_features_13,
            ],
            1,
        )
        x_64 = torch.nn.functional.batch_norm(
            concated_features_14,
            l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_14 = l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm1_parameters_bias_ = (None)
        x_65 = torch.nn.functional.relu(x_64, inplace=True)
        x_64 = None
        bottleneck_output_14 = torch.conv2d(
            x_65,
            l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_conv1_parameters_weight_ = (None)
        x_66 = torch.nn.functional.batch_norm(
            bottleneck_output_14,
            l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_14 = l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_norm2_parameters_bias_ = (None)
        x_67 = torch.nn.functional.relu(x_66, inplace=True)
        x_66 = None
        new_features_14 = torch.conv2d(
            x_67,
            l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_conv2_parameters_weight_ = (None)
        concated_features_15 = torch.cat(
            [
                input_7,
                new_features_6,
                new_features_7,
                new_features_8,
                new_features_9,
                new_features_10,
                new_features_11,
                new_features_12,
                new_features_13,
                new_features_14,
            ],
            1,
        )
        x_68 = torch.nn.functional.batch_norm(
            concated_features_15,
            l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_15 = l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm1_parameters_bias_ = (None)
        x_69 = torch.nn.functional.relu(x_68, inplace=True)
        x_68 = None
        bottleneck_output_15 = torch.conv2d(
            x_69,
            l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_69 = l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_conv1_parameters_weight_ = (None)
        x_70 = torch.nn.functional.batch_norm(
            bottleneck_output_15,
            l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_15 = l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_norm2_parameters_bias_ = (None)
        x_71 = torch.nn.functional.relu(x_70, inplace=True)
        x_70 = None
        new_features_15 = torch.conv2d(
            x_71,
            l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_71 = l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_conv2_parameters_weight_ = (None)
        concated_features_16 = torch.cat(
            [
                input_7,
                new_features_6,
                new_features_7,
                new_features_8,
                new_features_9,
                new_features_10,
                new_features_11,
                new_features_12,
                new_features_13,
                new_features_14,
                new_features_15,
            ],
            1,
        )
        x_72 = torch.nn.functional.batch_norm(
            concated_features_16,
            l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_16 = l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm1_parameters_bias_ = (None)
        x_73 = torch.nn.functional.relu(x_72, inplace=True)
        x_72 = None
        bottleneck_output_16 = torch.conv2d(
            x_73,
            l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_73 = l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_conv1_parameters_weight_ = (None)
        x_74 = torch.nn.functional.batch_norm(
            bottleneck_output_16,
            l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_16 = l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_norm2_parameters_bias_ = (None)
        x_75 = torch.nn.functional.relu(x_74, inplace=True)
        x_74 = None
        new_features_16 = torch.conv2d(
            x_75,
            l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_75 = l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_conv2_parameters_weight_ = (None)
        concated_features_17 = torch.cat(
            [
                input_7,
                new_features_6,
                new_features_7,
                new_features_8,
                new_features_9,
                new_features_10,
                new_features_11,
                new_features_12,
                new_features_13,
                new_features_14,
                new_features_15,
                new_features_16,
            ],
            1,
        )
        x_76 = torch.nn.functional.batch_norm(
            concated_features_17,
            l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_17 = l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm1_parameters_bias_ = (None)
        x_77 = torch.nn.functional.relu(x_76, inplace=True)
        x_76 = None
        bottleneck_output_17 = torch.conv2d(
            x_77,
            l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_77 = l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_conv1_parameters_weight_ = (None)
        x_78 = torch.nn.functional.batch_norm(
            bottleneck_output_17,
            l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_17 = l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_norm2_parameters_bias_ = (None)
        x_79 = torch.nn.functional.relu(x_78, inplace=True)
        x_78 = None
        new_features_17 = torch.conv2d(
            x_79,
            l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_conv2_parameters_weight_ = (None)
        input_8 = torch.cat(
            [
                input_7,
                new_features_6,
                new_features_7,
                new_features_8,
                new_features_9,
                new_features_10,
                new_features_11,
                new_features_12,
                new_features_13,
                new_features_14,
                new_features_15,
                new_features_16,
                new_features_17,
            ],
            1,
        )
        input_7 = (
            new_features_6
        ) = (
            new_features_7
        ) = (
            new_features_8
        ) = (
            new_features_9
        ) = (
            new_features_10
        ) = (
            new_features_11
        ) = (
            new_features_12
        ) = (
            new_features_13
        ) = new_features_14 = new_features_15 = new_features_16 = new_features_17 = None
        x_80 = torch.nn.functional.batch_norm(
            input_8,
            l_self_modules_features_modules_transition2_modules_norm_buffers_running_mean_,
            l_self_modules_features_modules_transition2_modules_norm_buffers_running_var_,
            l_self_modules_features_modules_transition2_modules_norm_parameters_weight_,
            l_self_modules_features_modules_transition2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_8 = l_self_modules_features_modules_transition2_modules_norm_buffers_running_mean_ = l_self_modules_features_modules_transition2_modules_norm_buffers_running_var_ = (
            l_self_modules_features_modules_transition2_modules_norm_parameters_weight_
        ) = (
            l_self_modules_features_modules_transition2_modules_norm_parameters_bias_
        ) = None
        x_81 = torch.nn.functional.relu(x_80, inplace=True)
        x_80 = None
        input_9 = torch.conv2d(
            x_81,
            l_self_modules_features_modules_transition2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_81 = (
            l_self_modules_features_modules_transition2_modules_conv_parameters_weight_
        ) = None
        input_10 = torch._C._nn.avg_pool2d(input_9, 2, 2, 0, False, True, None)
        input_9 = None
        concated_features_18 = torch.cat([input_10], 1)
        x_82 = torch.nn.functional.batch_norm(
            concated_features_18,
            l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_18 = l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm1_parameters_bias_ = (None)
        x_83 = torch.nn.functional.relu(x_82, inplace=True)
        x_82 = None
        bottleneck_output_18 = torch.conv2d(
            x_83,
            l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_83 = l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_conv1_parameters_weight_ = (None)
        x_84 = torch.nn.functional.batch_norm(
            bottleneck_output_18,
            l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_18 = l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_norm2_parameters_bias_ = (None)
        x_85 = torch.nn.functional.relu(x_84, inplace=True)
        x_84 = None
        new_features_18 = torch.conv2d(
            x_85,
            l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_85 = l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_conv2_parameters_weight_ = (None)
        concated_features_19 = torch.cat([input_10, new_features_18], 1)
        x_86 = torch.nn.functional.batch_norm(
            concated_features_19,
            l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_19 = l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm1_parameters_bias_ = (None)
        x_87 = torch.nn.functional.relu(x_86, inplace=True)
        x_86 = None
        bottleneck_output_19 = torch.conv2d(
            x_87,
            l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_87 = l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_conv1_parameters_weight_ = (None)
        x_88 = torch.nn.functional.batch_norm(
            bottleneck_output_19,
            l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_19 = l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_norm2_parameters_bias_ = (None)
        x_89 = torch.nn.functional.relu(x_88, inplace=True)
        x_88 = None
        new_features_19 = torch.conv2d(
            x_89,
            l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_conv2_parameters_weight_ = (None)
        concated_features_20 = torch.cat(
            [input_10, new_features_18, new_features_19], 1
        )
        x_90 = torch.nn.functional.batch_norm(
            concated_features_20,
            l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_20 = l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm1_parameters_bias_ = (None)
        x_91 = torch.nn.functional.relu(x_90, inplace=True)
        x_90 = None
        bottleneck_output_20 = torch.conv2d(
            x_91,
            l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_91 = l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_conv1_parameters_weight_ = (None)
        x_92 = torch.nn.functional.batch_norm(
            bottleneck_output_20,
            l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_20 = l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_norm2_parameters_bias_ = (None)
        x_93 = torch.nn.functional.relu(x_92, inplace=True)
        x_92 = None
        new_features_20 = torch.conv2d(
            x_93,
            l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_93 = l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_conv2_parameters_weight_ = (None)
        concated_features_21 = torch.cat(
            [input_10, new_features_18, new_features_19, new_features_20], 1
        )
        x_94 = torch.nn.functional.batch_norm(
            concated_features_21,
            l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_21 = l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm1_parameters_bias_ = (None)
        x_95 = torch.nn.functional.relu(x_94, inplace=True)
        x_94 = None
        bottleneck_output_21 = torch.conv2d(
            x_95,
            l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_95 = l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_conv1_parameters_weight_ = (None)
        x_96 = torch.nn.functional.batch_norm(
            bottleneck_output_21,
            l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_21 = l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_norm2_parameters_bias_ = (None)
        x_97 = torch.nn.functional.relu(x_96, inplace=True)
        x_96 = None
        new_features_21 = torch.conv2d(
            x_97,
            l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_conv2_parameters_weight_ = (None)
        concated_features_22 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
            ],
            1,
        )
        x_98 = torch.nn.functional.batch_norm(
            concated_features_22,
            l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_22 = l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm1_parameters_bias_ = (None)
        x_99 = torch.nn.functional.relu(x_98, inplace=True)
        x_98 = None
        bottleneck_output_22 = torch.conv2d(
            x_99,
            l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_99 = l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_conv1_parameters_weight_ = (None)
        x_100 = torch.nn.functional.batch_norm(
            bottleneck_output_22,
            l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_22 = l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_norm2_parameters_bias_ = (None)
        x_101 = torch.nn.functional.relu(x_100, inplace=True)
        x_100 = None
        new_features_22 = torch.conv2d(
            x_101,
            l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_101 = l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_conv2_parameters_weight_ = (None)
        concated_features_23 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
            ],
            1,
        )
        x_102 = torch.nn.functional.batch_norm(
            concated_features_23,
            l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_23 = l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm1_parameters_bias_ = (None)
        x_103 = torch.nn.functional.relu(x_102, inplace=True)
        x_102 = None
        bottleneck_output_23 = torch.conv2d(
            x_103,
            l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_conv1_parameters_weight_ = (None)
        x_104 = torch.nn.functional.batch_norm(
            bottleneck_output_23,
            l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_23 = l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_norm2_parameters_bias_ = (None)
        x_105 = torch.nn.functional.relu(x_104, inplace=True)
        x_104 = None
        new_features_23 = torch.conv2d(
            x_105,
            l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_105 = l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_conv2_parameters_weight_ = (None)
        concated_features_24 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
            ],
            1,
        )
        x_106 = torch.nn.functional.batch_norm(
            concated_features_24,
            l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_24 = l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm1_parameters_bias_ = (None)
        x_107 = torch.nn.functional.relu(x_106, inplace=True)
        x_106 = None
        bottleneck_output_24 = torch.conv2d(
            x_107,
            l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_107 = l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_conv1_parameters_weight_ = (None)
        x_108 = torch.nn.functional.batch_norm(
            bottleneck_output_24,
            l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_24 = l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_norm2_parameters_bias_ = (None)
        x_109 = torch.nn.functional.relu(x_108, inplace=True)
        x_108 = None
        new_features_24 = torch.conv2d(
            x_109,
            l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_conv2_parameters_weight_ = (None)
        concated_features_25 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
            ],
            1,
        )
        x_110 = torch.nn.functional.batch_norm(
            concated_features_25,
            l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_25 = l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm1_parameters_bias_ = (None)
        x_111 = torch.nn.functional.relu(x_110, inplace=True)
        x_110 = None
        bottleneck_output_25 = torch.conv2d(
            x_111,
            l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_111 = l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_conv1_parameters_weight_ = (None)
        x_112 = torch.nn.functional.batch_norm(
            bottleneck_output_25,
            l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_25 = l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_norm2_parameters_bias_ = (None)
        x_113 = torch.nn.functional.relu(x_112, inplace=True)
        x_112 = None
        new_features_25 = torch.conv2d(
            x_113,
            l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_113 = l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_conv2_parameters_weight_ = (None)
        concated_features_26 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
            ],
            1,
        )
        x_114 = torch.nn.functional.batch_norm(
            concated_features_26,
            l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_26 = l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm1_parameters_bias_ = (None)
        x_115 = torch.nn.functional.relu(x_114, inplace=True)
        x_114 = None
        bottleneck_output_26 = torch.conv2d(
            x_115,
            l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_115 = l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_conv1_parameters_weight_ = (None)
        x_116 = torch.nn.functional.batch_norm(
            bottleneck_output_26,
            l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_26 = l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_norm2_parameters_bias_ = (None)
        x_117 = torch.nn.functional.relu(x_116, inplace=True)
        x_116 = None
        new_features_26 = torch.conv2d(
            x_117,
            l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_conv2_parameters_weight_ = (None)
        concated_features_27 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
            ],
            1,
        )
        x_118 = torch.nn.functional.batch_norm(
            concated_features_27,
            l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_27 = l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm1_parameters_bias_ = (None)
        x_119 = torch.nn.functional.relu(x_118, inplace=True)
        x_118 = None
        bottleneck_output_27 = torch.conv2d(
            x_119,
            l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_119 = l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_conv1_parameters_weight_ = (None)
        x_120 = torch.nn.functional.batch_norm(
            bottleneck_output_27,
            l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_27 = l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_norm2_parameters_bias_ = (None)
        x_121 = torch.nn.functional.relu(x_120, inplace=True)
        x_120 = None
        new_features_27 = torch.conv2d(
            x_121,
            l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_121 = l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_conv2_parameters_weight_ = (None)
        concated_features_28 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
            ],
            1,
        )
        x_122 = torch.nn.functional.batch_norm(
            concated_features_28,
            l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_28 = l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm1_parameters_bias_ = (None)
        x_123 = torch.nn.functional.relu(x_122, inplace=True)
        x_122 = None
        bottleneck_output_28 = torch.conv2d(
            x_123,
            l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_123 = l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_conv1_parameters_weight_ = (None)
        x_124 = torch.nn.functional.batch_norm(
            bottleneck_output_28,
            l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_28 = l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_norm2_parameters_bias_ = (None)
        x_125 = torch.nn.functional.relu(x_124, inplace=True)
        x_124 = None
        new_features_28 = torch.conv2d(
            x_125,
            l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_conv2_parameters_weight_ = (None)
        concated_features_29 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
            ],
            1,
        )
        x_126 = torch.nn.functional.batch_norm(
            concated_features_29,
            l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_29 = l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm1_parameters_bias_ = (None)
        x_127 = torch.nn.functional.relu(x_126, inplace=True)
        x_126 = None
        bottleneck_output_29 = torch.conv2d(
            x_127,
            l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_127 = l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_conv1_parameters_weight_ = (None)
        x_128 = torch.nn.functional.batch_norm(
            bottleneck_output_29,
            l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_29 = l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_norm2_parameters_bias_ = (None)
        x_129 = torch.nn.functional.relu(x_128, inplace=True)
        x_128 = None
        new_features_29 = torch.conv2d(
            x_129,
            l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_129 = l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_conv2_parameters_weight_ = (None)
        concated_features_30 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
            ],
            1,
        )
        x_130 = torch.nn.functional.batch_norm(
            concated_features_30,
            l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_30 = l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm1_parameters_bias_ = (None)
        x_131 = torch.nn.functional.relu(x_130, inplace=True)
        x_130 = None
        bottleneck_output_30 = torch.conv2d(
            x_131,
            l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_131 = l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_conv1_parameters_weight_ = (None)
        x_132 = torch.nn.functional.batch_norm(
            bottleneck_output_30,
            l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_30 = l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_norm2_parameters_bias_ = (None)
        x_133 = torch.nn.functional.relu(x_132, inplace=True)
        x_132 = None
        new_features_30 = torch.conv2d(
            x_133,
            l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_133 = l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_conv2_parameters_weight_ = (None)
        concated_features_31 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
            ],
            1,
        )
        x_134 = torch.nn.functional.batch_norm(
            concated_features_31,
            l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_31 = l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm1_parameters_bias_ = (None)
        x_135 = torch.nn.functional.relu(x_134, inplace=True)
        x_134 = None
        bottleneck_output_31 = torch.conv2d(
            x_135,
            l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_135 = l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_conv1_parameters_weight_ = (None)
        x_136 = torch.nn.functional.batch_norm(
            bottleneck_output_31,
            l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_31 = l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_norm2_parameters_bias_ = (None)
        x_137 = torch.nn.functional.relu(x_136, inplace=True)
        x_136 = None
        new_features_31 = torch.conv2d(
            x_137,
            l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_137 = l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_conv2_parameters_weight_ = (None)
        concated_features_32 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
            ],
            1,
        )
        x_138 = torch.nn.functional.batch_norm(
            concated_features_32,
            l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_32 = l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm1_parameters_bias_ = (None)
        x_139 = torch.nn.functional.relu(x_138, inplace=True)
        x_138 = None
        bottleneck_output_32 = torch.conv2d(
            x_139,
            l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_conv1_parameters_weight_ = (None)
        x_140 = torch.nn.functional.batch_norm(
            bottleneck_output_32,
            l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_32 = l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_norm2_parameters_bias_ = (None)
        x_141 = torch.nn.functional.relu(x_140, inplace=True)
        x_140 = None
        new_features_32 = torch.conv2d(
            x_141,
            l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_141 = l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_conv2_parameters_weight_ = (None)
        concated_features_33 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
            ],
            1,
        )
        x_142 = torch.nn.functional.batch_norm(
            concated_features_33,
            l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_33 = l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm1_parameters_bias_ = (None)
        x_143 = torch.nn.functional.relu(x_142, inplace=True)
        x_142 = None
        bottleneck_output_33 = torch.conv2d(
            x_143,
            l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_143 = l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_conv1_parameters_weight_ = (None)
        x_144 = torch.nn.functional.batch_norm(
            bottleneck_output_33,
            l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_33 = l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_norm2_parameters_bias_ = (None)
        x_145 = torch.nn.functional.relu(x_144, inplace=True)
        x_144 = None
        new_features_33 = torch.conv2d(
            x_145,
            l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_145 = l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_conv2_parameters_weight_ = (None)
        concated_features_34 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
            ],
            1,
        )
        x_146 = torch.nn.functional.batch_norm(
            concated_features_34,
            l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_34 = l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm1_parameters_bias_ = (None)
        x_147 = torch.nn.functional.relu(x_146, inplace=True)
        x_146 = None
        bottleneck_output_34 = torch.conv2d(
            x_147,
            l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_147 = l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_conv1_parameters_weight_ = (None)
        x_148 = torch.nn.functional.batch_norm(
            bottleneck_output_34,
            l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_34 = l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_norm2_parameters_bias_ = (None)
        x_149 = torch.nn.functional.relu(x_148, inplace=True)
        x_148 = None
        new_features_34 = torch.conv2d(
            x_149,
            l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_149 = l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_conv2_parameters_weight_ = (None)
        concated_features_35 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
            ],
            1,
        )
        x_150 = torch.nn.functional.batch_norm(
            concated_features_35,
            l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_35 = l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm1_parameters_bias_ = (None)
        x_151 = torch.nn.functional.relu(x_150, inplace=True)
        x_150 = None
        bottleneck_output_35 = torch.conv2d(
            x_151,
            l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_151 = l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_conv1_parameters_weight_ = (None)
        x_152 = torch.nn.functional.batch_norm(
            bottleneck_output_35,
            l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_35 = l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_norm2_parameters_bias_ = (None)
        x_153 = torch.nn.functional.relu(x_152, inplace=True)
        x_152 = None
        new_features_35 = torch.conv2d(
            x_153,
            l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_153 = l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_conv2_parameters_weight_ = (None)
        concated_features_36 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
            ],
            1,
        )
        x_154 = torch.nn.functional.batch_norm(
            concated_features_36,
            l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_36 = l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm1_parameters_bias_ = (None)
        x_155 = torch.nn.functional.relu(x_154, inplace=True)
        x_154 = None
        bottleneck_output_36 = torch.conv2d(
            x_155,
            l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_155 = l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_conv1_parameters_weight_ = (None)
        x_156 = torch.nn.functional.batch_norm(
            bottleneck_output_36,
            l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_36 = l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_norm2_parameters_bias_ = (None)
        x_157 = torch.nn.functional.relu(x_156, inplace=True)
        x_156 = None
        new_features_36 = torch.conv2d(
            x_157,
            l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_157 = l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_conv2_parameters_weight_ = (None)
        concated_features_37 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
            ],
            1,
        )
        x_158 = torch.nn.functional.batch_norm(
            concated_features_37,
            l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_37 = l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm1_parameters_bias_ = (None)
        x_159 = torch.nn.functional.relu(x_158, inplace=True)
        x_158 = None
        bottleneck_output_37 = torch.conv2d(
            x_159,
            l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_159 = l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_conv1_parameters_weight_ = (None)
        x_160 = torch.nn.functional.batch_norm(
            bottleneck_output_37,
            l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_37 = l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_norm2_parameters_bias_ = (None)
        x_161 = torch.nn.functional.relu(x_160, inplace=True)
        x_160 = None
        new_features_37 = torch.conv2d(
            x_161,
            l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_161 = l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_conv2_parameters_weight_ = (None)
        concated_features_38 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
            ],
            1,
        )
        x_162 = torch.nn.functional.batch_norm(
            concated_features_38,
            l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_38 = l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm1_parameters_bias_ = (None)
        x_163 = torch.nn.functional.relu(x_162, inplace=True)
        x_162 = None
        bottleneck_output_38 = torch.conv2d(
            x_163,
            l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_163 = l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_conv1_parameters_weight_ = (None)
        x_164 = torch.nn.functional.batch_norm(
            bottleneck_output_38,
            l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_38 = l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_norm2_parameters_bias_ = (None)
        x_165 = torch.nn.functional.relu(x_164, inplace=True)
        x_164 = None
        new_features_38 = torch.conv2d(
            x_165,
            l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_165 = l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_conv2_parameters_weight_ = (None)
        concated_features_39 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
            ],
            1,
        )
        x_166 = torch.nn.functional.batch_norm(
            concated_features_39,
            l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_39 = l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm1_parameters_bias_ = (None)
        x_167 = torch.nn.functional.relu(x_166, inplace=True)
        x_166 = None
        bottleneck_output_39 = torch.conv2d(
            x_167,
            l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_167 = l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_conv1_parameters_weight_ = (None)
        x_168 = torch.nn.functional.batch_norm(
            bottleneck_output_39,
            l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_39 = l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_norm2_parameters_bias_ = (None)
        x_169 = torch.nn.functional.relu(x_168, inplace=True)
        x_168 = None
        new_features_39 = torch.conv2d(
            x_169,
            l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_169 = l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_conv2_parameters_weight_ = (None)
        concated_features_40 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
            ],
            1,
        )
        x_170 = torch.nn.functional.batch_norm(
            concated_features_40,
            l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_40 = l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm1_parameters_bias_ = (None)
        x_171 = torch.nn.functional.relu(x_170, inplace=True)
        x_170 = None
        bottleneck_output_40 = torch.conv2d(
            x_171,
            l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_171 = l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_conv1_parameters_weight_ = (None)
        x_172 = torch.nn.functional.batch_norm(
            bottleneck_output_40,
            l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_40 = l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_norm2_parameters_bias_ = (None)
        x_173 = torch.nn.functional.relu(x_172, inplace=True)
        x_172 = None
        new_features_40 = torch.conv2d(
            x_173,
            l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_173 = l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_conv2_parameters_weight_ = (None)
        concated_features_41 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
            ],
            1,
        )
        x_174 = torch.nn.functional.batch_norm(
            concated_features_41,
            l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_41 = l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm1_parameters_bias_ = (None)
        x_175 = torch.nn.functional.relu(x_174, inplace=True)
        x_174 = None
        bottleneck_output_41 = torch.conv2d(
            x_175,
            l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_175 = l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_conv1_parameters_weight_ = (None)
        x_176 = torch.nn.functional.batch_norm(
            bottleneck_output_41,
            l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_41 = l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_norm2_parameters_bias_ = (None)
        x_177 = torch.nn.functional.relu(x_176, inplace=True)
        x_176 = None
        new_features_41 = torch.conv2d(
            x_177,
            l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_177 = l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_conv2_parameters_weight_ = (None)
        concated_features_42 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
            ],
            1,
        )
        x_178 = torch.nn.functional.batch_norm(
            concated_features_42,
            l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_42 = l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm1_parameters_bias_ = (None)
        x_179 = torch.nn.functional.relu(x_178, inplace=True)
        x_178 = None
        bottleneck_output_42 = torch.conv2d(
            x_179,
            l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_179 = l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_conv1_parameters_weight_ = (None)
        x_180 = torch.nn.functional.batch_norm(
            bottleneck_output_42,
            l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_42 = l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_norm2_parameters_bias_ = (None)
        x_181 = torch.nn.functional.relu(x_180, inplace=True)
        x_180 = None
        new_features_42 = torch.conv2d(
            x_181,
            l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_181 = l_self_modules_features_modules_denseblock3_modules_denselayer25_modules_conv2_parameters_weight_ = (None)
        concated_features_43 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
            ],
            1,
        )
        x_182 = torch.nn.functional.batch_norm(
            concated_features_43,
            l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_43 = l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm1_parameters_bias_ = (None)
        x_183 = torch.nn.functional.relu(x_182, inplace=True)
        x_182 = None
        bottleneck_output_43 = torch.conv2d(
            x_183,
            l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_183 = l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_conv1_parameters_weight_ = (None)
        x_184 = torch.nn.functional.batch_norm(
            bottleneck_output_43,
            l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_43 = l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_norm2_parameters_bias_ = (None)
        x_185 = torch.nn.functional.relu(x_184, inplace=True)
        x_184 = None
        new_features_43 = torch.conv2d(
            x_185,
            l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_185 = l_self_modules_features_modules_denseblock3_modules_denselayer26_modules_conv2_parameters_weight_ = (None)
        concated_features_44 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
            ],
            1,
        )
        x_186 = torch.nn.functional.batch_norm(
            concated_features_44,
            l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_44 = l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm1_parameters_bias_ = (None)
        x_187 = torch.nn.functional.relu(x_186, inplace=True)
        x_186 = None
        bottleneck_output_44 = torch.conv2d(
            x_187,
            l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_187 = l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_conv1_parameters_weight_ = (None)
        x_188 = torch.nn.functional.batch_norm(
            bottleneck_output_44,
            l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_44 = l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_norm2_parameters_bias_ = (None)
        x_189 = torch.nn.functional.relu(x_188, inplace=True)
        x_188 = None
        new_features_44 = torch.conv2d(
            x_189,
            l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_189 = l_self_modules_features_modules_denseblock3_modules_denselayer27_modules_conv2_parameters_weight_ = (None)
        concated_features_45 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
            ],
            1,
        )
        x_190 = torch.nn.functional.batch_norm(
            concated_features_45,
            l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_45 = l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm1_parameters_bias_ = (None)
        x_191 = torch.nn.functional.relu(x_190, inplace=True)
        x_190 = None
        bottleneck_output_45 = torch.conv2d(
            x_191,
            l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_191 = l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_conv1_parameters_weight_ = (None)
        x_192 = torch.nn.functional.batch_norm(
            bottleneck_output_45,
            l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_45 = l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_norm2_parameters_bias_ = (None)
        x_193 = torch.nn.functional.relu(x_192, inplace=True)
        x_192 = None
        new_features_45 = torch.conv2d(
            x_193,
            l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_193 = l_self_modules_features_modules_denseblock3_modules_denselayer28_modules_conv2_parameters_weight_ = (None)
        concated_features_46 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
            ],
            1,
        )
        x_194 = torch.nn.functional.batch_norm(
            concated_features_46,
            l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_46 = l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm1_parameters_bias_ = (None)
        x_195 = torch.nn.functional.relu(x_194, inplace=True)
        x_194 = None
        bottleneck_output_46 = torch.conv2d(
            x_195,
            l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_195 = l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_conv1_parameters_weight_ = (None)
        x_196 = torch.nn.functional.batch_norm(
            bottleneck_output_46,
            l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_46 = l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_norm2_parameters_bias_ = (None)
        x_197 = torch.nn.functional.relu(x_196, inplace=True)
        x_196 = None
        new_features_46 = torch.conv2d(
            x_197,
            l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_197 = l_self_modules_features_modules_denseblock3_modules_denselayer29_modules_conv2_parameters_weight_ = (None)
        concated_features_47 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
            ],
            1,
        )
        x_198 = torch.nn.functional.batch_norm(
            concated_features_47,
            l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_47 = l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm1_parameters_bias_ = (None)
        x_199 = torch.nn.functional.relu(x_198, inplace=True)
        x_198 = None
        bottleneck_output_47 = torch.conv2d(
            x_199,
            l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_199 = l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_conv1_parameters_weight_ = (None)
        x_200 = torch.nn.functional.batch_norm(
            bottleneck_output_47,
            l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_47 = l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_norm2_parameters_bias_ = (None)
        x_201 = torch.nn.functional.relu(x_200, inplace=True)
        x_200 = None
        new_features_47 = torch.conv2d(
            x_201,
            l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_201 = l_self_modules_features_modules_denseblock3_modules_denselayer30_modules_conv2_parameters_weight_ = (None)
        concated_features_48 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
            ],
            1,
        )
        x_202 = torch.nn.functional.batch_norm(
            concated_features_48,
            l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_48 = l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm1_parameters_bias_ = (None)
        x_203 = torch.nn.functional.relu(x_202, inplace=True)
        x_202 = None
        bottleneck_output_48 = torch.conv2d(
            x_203,
            l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_203 = l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_conv1_parameters_weight_ = (None)
        x_204 = torch.nn.functional.batch_norm(
            bottleneck_output_48,
            l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_48 = l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_norm2_parameters_bias_ = (None)
        x_205 = torch.nn.functional.relu(x_204, inplace=True)
        x_204 = None
        new_features_48 = torch.conv2d(
            x_205,
            l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_205 = l_self_modules_features_modules_denseblock3_modules_denselayer31_modules_conv2_parameters_weight_ = (None)
        concated_features_49 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
            ],
            1,
        )
        x_206 = torch.nn.functional.batch_norm(
            concated_features_49,
            l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_49 = l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm1_parameters_bias_ = (None)
        x_207 = torch.nn.functional.relu(x_206, inplace=True)
        x_206 = None
        bottleneck_output_49 = torch.conv2d(
            x_207,
            l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_207 = l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_conv1_parameters_weight_ = (None)
        x_208 = torch.nn.functional.batch_norm(
            bottleneck_output_49,
            l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_49 = l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_norm2_parameters_bias_ = (None)
        x_209 = torch.nn.functional.relu(x_208, inplace=True)
        x_208 = None
        new_features_49 = torch.conv2d(
            x_209,
            l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_209 = l_self_modules_features_modules_denseblock3_modules_denselayer32_modules_conv2_parameters_weight_ = (None)
        concated_features_50 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
            ],
            1,
        )
        x_210 = torch.nn.functional.batch_norm(
            concated_features_50,
            l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_50 = l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm1_parameters_bias_ = (None)
        x_211 = torch.nn.functional.relu(x_210, inplace=True)
        x_210 = None
        bottleneck_output_50 = torch.conv2d(
            x_211,
            l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_211 = l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_conv1_parameters_weight_ = (None)
        x_212 = torch.nn.functional.batch_norm(
            bottleneck_output_50,
            l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_50 = l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_norm2_parameters_bias_ = (None)
        x_213 = torch.nn.functional.relu(x_212, inplace=True)
        x_212 = None
        new_features_50 = torch.conv2d(
            x_213,
            l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_213 = l_self_modules_features_modules_denseblock3_modules_denselayer33_modules_conv2_parameters_weight_ = (None)
        concated_features_51 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
            ],
            1,
        )
        x_214 = torch.nn.functional.batch_norm(
            concated_features_51,
            l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_51 = l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm1_parameters_bias_ = (None)
        x_215 = torch.nn.functional.relu(x_214, inplace=True)
        x_214 = None
        bottleneck_output_51 = torch.conv2d(
            x_215,
            l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_215 = l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_conv1_parameters_weight_ = (None)
        x_216 = torch.nn.functional.batch_norm(
            bottleneck_output_51,
            l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_51 = l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_norm2_parameters_bias_ = (None)
        x_217 = torch.nn.functional.relu(x_216, inplace=True)
        x_216 = None
        new_features_51 = torch.conv2d(
            x_217,
            l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_217 = l_self_modules_features_modules_denseblock3_modules_denselayer34_modules_conv2_parameters_weight_ = (None)
        concated_features_52 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
            ],
            1,
        )
        x_218 = torch.nn.functional.batch_norm(
            concated_features_52,
            l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_52 = l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm1_parameters_bias_ = (None)
        x_219 = torch.nn.functional.relu(x_218, inplace=True)
        x_218 = None
        bottleneck_output_52 = torch.conv2d(
            x_219,
            l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_219 = l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_conv1_parameters_weight_ = (None)
        x_220 = torch.nn.functional.batch_norm(
            bottleneck_output_52,
            l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_52 = l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_norm2_parameters_bias_ = (None)
        x_221 = torch.nn.functional.relu(x_220, inplace=True)
        x_220 = None
        new_features_52 = torch.conv2d(
            x_221,
            l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_221 = l_self_modules_features_modules_denseblock3_modules_denselayer35_modules_conv2_parameters_weight_ = (None)
        concated_features_53 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
            ],
            1,
        )
        x_222 = torch.nn.functional.batch_norm(
            concated_features_53,
            l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_53 = l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm1_parameters_bias_ = (None)
        x_223 = torch.nn.functional.relu(x_222, inplace=True)
        x_222 = None
        bottleneck_output_53 = torch.conv2d(
            x_223,
            l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_223 = l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_conv1_parameters_weight_ = (None)
        x_224 = torch.nn.functional.batch_norm(
            bottleneck_output_53,
            l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_53 = l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_norm2_parameters_bias_ = (None)
        x_225 = torch.nn.functional.relu(x_224, inplace=True)
        x_224 = None
        new_features_53 = torch.conv2d(
            x_225,
            l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_225 = l_self_modules_features_modules_denseblock3_modules_denselayer36_modules_conv2_parameters_weight_ = (None)
        concated_features_54 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
            ],
            1,
        )
        x_226 = torch.nn.functional.batch_norm(
            concated_features_54,
            l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_54 = l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm1_parameters_bias_ = (None)
        x_227 = torch.nn.functional.relu(x_226, inplace=True)
        x_226 = None
        bottleneck_output_54 = torch.conv2d(
            x_227,
            l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_227 = l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_conv1_parameters_weight_ = (None)
        x_228 = torch.nn.functional.batch_norm(
            bottleneck_output_54,
            l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_54 = l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_norm2_parameters_bias_ = (None)
        x_229 = torch.nn.functional.relu(x_228, inplace=True)
        x_228 = None
        new_features_54 = torch.conv2d(
            x_229,
            l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_229 = l_self_modules_features_modules_denseblock3_modules_denselayer37_modules_conv2_parameters_weight_ = (None)
        concated_features_55 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
            ],
            1,
        )
        x_230 = torch.nn.functional.batch_norm(
            concated_features_55,
            l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_55 = l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm1_parameters_bias_ = (None)
        x_231 = torch.nn.functional.relu(x_230, inplace=True)
        x_230 = None
        bottleneck_output_55 = torch.conv2d(
            x_231,
            l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_231 = l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_conv1_parameters_weight_ = (None)
        x_232 = torch.nn.functional.batch_norm(
            bottleneck_output_55,
            l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_55 = l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_norm2_parameters_bias_ = (None)
        x_233 = torch.nn.functional.relu(x_232, inplace=True)
        x_232 = None
        new_features_55 = torch.conv2d(
            x_233,
            l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_233 = l_self_modules_features_modules_denseblock3_modules_denselayer38_modules_conv2_parameters_weight_ = (None)
        concated_features_56 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
            ],
            1,
        )
        x_234 = torch.nn.functional.batch_norm(
            concated_features_56,
            l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_56 = l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm1_parameters_bias_ = (None)
        x_235 = torch.nn.functional.relu(x_234, inplace=True)
        x_234 = None
        bottleneck_output_56 = torch.conv2d(
            x_235,
            l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_235 = l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_conv1_parameters_weight_ = (None)
        x_236 = torch.nn.functional.batch_norm(
            bottleneck_output_56,
            l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_56 = l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_norm2_parameters_bias_ = (None)
        x_237 = torch.nn.functional.relu(x_236, inplace=True)
        x_236 = None
        new_features_56 = torch.conv2d(
            x_237,
            l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_237 = l_self_modules_features_modules_denseblock3_modules_denselayer39_modules_conv2_parameters_weight_ = (None)
        concated_features_57 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
            ],
            1,
        )
        x_238 = torch.nn.functional.batch_norm(
            concated_features_57,
            l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_57 = l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm1_parameters_bias_ = (None)
        x_239 = torch.nn.functional.relu(x_238, inplace=True)
        x_238 = None
        bottleneck_output_57 = torch.conv2d(
            x_239,
            l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_239 = l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_conv1_parameters_weight_ = (None)
        x_240 = torch.nn.functional.batch_norm(
            bottleneck_output_57,
            l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_57 = l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_norm2_parameters_bias_ = (None)
        x_241 = torch.nn.functional.relu(x_240, inplace=True)
        x_240 = None
        new_features_57 = torch.conv2d(
            x_241,
            l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_241 = l_self_modules_features_modules_denseblock3_modules_denselayer40_modules_conv2_parameters_weight_ = (None)
        concated_features_58 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
            ],
            1,
        )
        x_242 = torch.nn.functional.batch_norm(
            concated_features_58,
            l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_58 = l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm1_parameters_bias_ = (None)
        x_243 = torch.nn.functional.relu(x_242, inplace=True)
        x_242 = None
        bottleneck_output_58 = torch.conv2d(
            x_243,
            l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_243 = l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_conv1_parameters_weight_ = (None)
        x_244 = torch.nn.functional.batch_norm(
            bottleneck_output_58,
            l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_58 = l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_norm2_parameters_bias_ = (None)
        x_245 = torch.nn.functional.relu(x_244, inplace=True)
        x_244 = None
        new_features_58 = torch.conv2d(
            x_245,
            l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_245 = l_self_modules_features_modules_denseblock3_modules_denselayer41_modules_conv2_parameters_weight_ = (None)
        concated_features_59 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
            ],
            1,
        )
        x_246 = torch.nn.functional.batch_norm(
            concated_features_59,
            l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_59 = l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm1_parameters_bias_ = (None)
        x_247 = torch.nn.functional.relu(x_246, inplace=True)
        x_246 = None
        bottleneck_output_59 = torch.conv2d(
            x_247,
            l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_247 = l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_conv1_parameters_weight_ = (None)
        x_248 = torch.nn.functional.batch_norm(
            bottleneck_output_59,
            l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_59 = l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_norm2_parameters_bias_ = (None)
        x_249 = torch.nn.functional.relu(x_248, inplace=True)
        x_248 = None
        new_features_59 = torch.conv2d(
            x_249,
            l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_249 = l_self_modules_features_modules_denseblock3_modules_denselayer42_modules_conv2_parameters_weight_ = (None)
        concated_features_60 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
            ],
            1,
        )
        x_250 = torch.nn.functional.batch_norm(
            concated_features_60,
            l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_60 = l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm1_parameters_bias_ = (None)
        x_251 = torch.nn.functional.relu(x_250, inplace=True)
        x_250 = None
        bottleneck_output_60 = torch.conv2d(
            x_251,
            l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_251 = l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_conv1_parameters_weight_ = (None)
        x_252 = torch.nn.functional.batch_norm(
            bottleneck_output_60,
            l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_60 = l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_norm2_parameters_bias_ = (None)
        x_253 = torch.nn.functional.relu(x_252, inplace=True)
        x_252 = None
        new_features_60 = torch.conv2d(
            x_253,
            l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_253 = l_self_modules_features_modules_denseblock3_modules_denselayer43_modules_conv2_parameters_weight_ = (None)
        concated_features_61 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
            ],
            1,
        )
        x_254 = torch.nn.functional.batch_norm(
            concated_features_61,
            l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_61 = l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm1_parameters_bias_ = (None)
        x_255 = torch.nn.functional.relu(x_254, inplace=True)
        x_254 = None
        bottleneck_output_61 = torch.conv2d(
            x_255,
            l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_255 = l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_conv1_parameters_weight_ = (None)
        x_256 = torch.nn.functional.batch_norm(
            bottleneck_output_61,
            l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_61 = l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_norm2_parameters_bias_ = (None)
        x_257 = torch.nn.functional.relu(x_256, inplace=True)
        x_256 = None
        new_features_61 = torch.conv2d(
            x_257,
            l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_257 = l_self_modules_features_modules_denseblock3_modules_denselayer44_modules_conv2_parameters_weight_ = (None)
        concated_features_62 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
            ],
            1,
        )
        x_258 = torch.nn.functional.batch_norm(
            concated_features_62,
            l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_62 = l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm1_parameters_bias_ = (None)
        x_259 = torch.nn.functional.relu(x_258, inplace=True)
        x_258 = None
        bottleneck_output_62 = torch.conv2d(
            x_259,
            l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_259 = l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_conv1_parameters_weight_ = (None)
        x_260 = torch.nn.functional.batch_norm(
            bottleneck_output_62,
            l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_62 = l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_norm2_parameters_bias_ = (None)
        x_261 = torch.nn.functional.relu(x_260, inplace=True)
        x_260 = None
        new_features_62 = torch.conv2d(
            x_261,
            l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_261 = l_self_modules_features_modules_denseblock3_modules_denselayer45_modules_conv2_parameters_weight_ = (None)
        concated_features_63 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
            ],
            1,
        )
        x_262 = torch.nn.functional.batch_norm(
            concated_features_63,
            l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_63 = l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm1_parameters_bias_ = (None)
        x_263 = torch.nn.functional.relu(x_262, inplace=True)
        x_262 = None
        bottleneck_output_63 = torch.conv2d(
            x_263,
            l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_263 = l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_conv1_parameters_weight_ = (None)
        x_264 = torch.nn.functional.batch_norm(
            bottleneck_output_63,
            l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_63 = l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_norm2_parameters_bias_ = (None)
        x_265 = torch.nn.functional.relu(x_264, inplace=True)
        x_264 = None
        new_features_63 = torch.conv2d(
            x_265,
            l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_265 = l_self_modules_features_modules_denseblock3_modules_denselayer46_modules_conv2_parameters_weight_ = (None)
        concated_features_64 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
            ],
            1,
        )
        x_266 = torch.nn.functional.batch_norm(
            concated_features_64,
            l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_64 = l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm1_parameters_bias_ = (None)
        x_267 = torch.nn.functional.relu(x_266, inplace=True)
        x_266 = None
        bottleneck_output_64 = torch.conv2d(
            x_267,
            l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_267 = l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_conv1_parameters_weight_ = (None)
        x_268 = torch.nn.functional.batch_norm(
            bottleneck_output_64,
            l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_64 = l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_norm2_parameters_bias_ = (None)
        x_269 = torch.nn.functional.relu(x_268, inplace=True)
        x_268 = None
        new_features_64 = torch.conv2d(
            x_269,
            l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_269 = l_self_modules_features_modules_denseblock3_modules_denselayer47_modules_conv2_parameters_weight_ = (None)
        concated_features_65 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
                new_features_64,
            ],
            1,
        )
        x_270 = torch.nn.functional.batch_norm(
            concated_features_65,
            l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_65 = l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm1_parameters_bias_ = (None)
        x_271 = torch.nn.functional.relu(x_270, inplace=True)
        x_270 = None
        bottleneck_output_65 = torch.conv2d(
            x_271,
            l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_271 = l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_conv1_parameters_weight_ = (None)
        x_272 = torch.nn.functional.batch_norm(
            bottleneck_output_65,
            l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_65 = l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_norm2_parameters_bias_ = (None)
        x_273 = torch.nn.functional.relu(x_272, inplace=True)
        x_272 = None
        new_features_65 = torch.conv2d(
            x_273,
            l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_273 = l_self_modules_features_modules_denseblock3_modules_denselayer48_modules_conv2_parameters_weight_ = (None)
        concated_features_66 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
                new_features_64,
                new_features_65,
            ],
            1,
        )
        x_274 = torch.nn.functional.batch_norm(
            concated_features_66,
            l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_66 = l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm1_parameters_bias_ = (None)
        x_275 = torch.nn.functional.relu(x_274, inplace=True)
        x_274 = None
        bottleneck_output_66 = torch.conv2d(
            x_275,
            l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_275 = l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_conv1_parameters_weight_ = (None)
        x_276 = torch.nn.functional.batch_norm(
            bottleneck_output_66,
            l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_66 = l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_norm2_parameters_bias_ = (None)
        x_277 = torch.nn.functional.relu(x_276, inplace=True)
        x_276 = None
        new_features_66 = torch.conv2d(
            x_277,
            l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_277 = l_self_modules_features_modules_denseblock3_modules_denselayer49_modules_conv2_parameters_weight_ = (None)
        concated_features_67 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
                new_features_64,
                new_features_65,
                new_features_66,
            ],
            1,
        )
        x_278 = torch.nn.functional.batch_norm(
            concated_features_67,
            l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_67 = l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm1_parameters_bias_ = (None)
        x_279 = torch.nn.functional.relu(x_278, inplace=True)
        x_278 = None
        bottleneck_output_67 = torch.conv2d(
            x_279,
            l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_279 = l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_conv1_parameters_weight_ = (None)
        x_280 = torch.nn.functional.batch_norm(
            bottleneck_output_67,
            l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_67 = l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_norm2_parameters_bias_ = (None)
        x_281 = torch.nn.functional.relu(x_280, inplace=True)
        x_280 = None
        new_features_67 = torch.conv2d(
            x_281,
            l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_281 = l_self_modules_features_modules_denseblock3_modules_denselayer50_modules_conv2_parameters_weight_ = (None)
        concated_features_68 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
                new_features_64,
                new_features_65,
                new_features_66,
                new_features_67,
            ],
            1,
        )
        x_282 = torch.nn.functional.batch_norm(
            concated_features_68,
            l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_68 = l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm1_parameters_bias_ = (None)
        x_283 = torch.nn.functional.relu(x_282, inplace=True)
        x_282 = None
        bottleneck_output_68 = torch.conv2d(
            x_283,
            l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_283 = l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_conv1_parameters_weight_ = (None)
        x_284 = torch.nn.functional.batch_norm(
            bottleneck_output_68,
            l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_68 = l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_norm2_parameters_bias_ = (None)
        x_285 = torch.nn.functional.relu(x_284, inplace=True)
        x_284 = None
        new_features_68 = torch.conv2d(
            x_285,
            l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_285 = l_self_modules_features_modules_denseblock3_modules_denselayer51_modules_conv2_parameters_weight_ = (None)
        concated_features_69 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
                new_features_64,
                new_features_65,
                new_features_66,
                new_features_67,
                new_features_68,
            ],
            1,
        )
        x_286 = torch.nn.functional.batch_norm(
            concated_features_69,
            l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_69 = l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm1_parameters_bias_ = (None)
        x_287 = torch.nn.functional.relu(x_286, inplace=True)
        x_286 = None
        bottleneck_output_69 = torch.conv2d(
            x_287,
            l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_287 = l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_conv1_parameters_weight_ = (None)
        x_288 = torch.nn.functional.batch_norm(
            bottleneck_output_69,
            l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_69 = l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_norm2_parameters_bias_ = (None)
        x_289 = torch.nn.functional.relu(x_288, inplace=True)
        x_288 = None
        new_features_69 = torch.conv2d(
            x_289,
            l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_289 = l_self_modules_features_modules_denseblock3_modules_denselayer52_modules_conv2_parameters_weight_ = (None)
        concated_features_70 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
                new_features_64,
                new_features_65,
                new_features_66,
                new_features_67,
                new_features_68,
                new_features_69,
            ],
            1,
        )
        x_290 = torch.nn.functional.batch_norm(
            concated_features_70,
            l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_70 = l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm1_parameters_bias_ = (None)
        x_291 = torch.nn.functional.relu(x_290, inplace=True)
        x_290 = None
        bottleneck_output_70 = torch.conv2d(
            x_291,
            l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_291 = l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_conv1_parameters_weight_ = (None)
        x_292 = torch.nn.functional.batch_norm(
            bottleneck_output_70,
            l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_70 = l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_norm2_parameters_bias_ = (None)
        x_293 = torch.nn.functional.relu(x_292, inplace=True)
        x_292 = None
        new_features_70 = torch.conv2d(
            x_293,
            l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_293 = l_self_modules_features_modules_denseblock3_modules_denselayer53_modules_conv2_parameters_weight_ = (None)
        concated_features_71 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
                new_features_64,
                new_features_65,
                new_features_66,
                new_features_67,
                new_features_68,
                new_features_69,
                new_features_70,
            ],
            1,
        )
        x_294 = torch.nn.functional.batch_norm(
            concated_features_71,
            l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_71 = l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm1_parameters_bias_ = (None)
        x_295 = torch.nn.functional.relu(x_294, inplace=True)
        x_294 = None
        bottleneck_output_71 = torch.conv2d(
            x_295,
            l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_295 = l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_conv1_parameters_weight_ = (None)
        x_296 = torch.nn.functional.batch_norm(
            bottleneck_output_71,
            l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_71 = l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_norm2_parameters_bias_ = (None)
        x_297 = torch.nn.functional.relu(x_296, inplace=True)
        x_296 = None
        new_features_71 = torch.conv2d(
            x_297,
            l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_297 = l_self_modules_features_modules_denseblock3_modules_denselayer54_modules_conv2_parameters_weight_ = (None)
        concated_features_72 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
                new_features_64,
                new_features_65,
                new_features_66,
                new_features_67,
                new_features_68,
                new_features_69,
                new_features_70,
                new_features_71,
            ],
            1,
        )
        x_298 = torch.nn.functional.batch_norm(
            concated_features_72,
            l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_72 = l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm1_parameters_bias_ = (None)
        x_299 = torch.nn.functional.relu(x_298, inplace=True)
        x_298 = None
        bottleneck_output_72 = torch.conv2d(
            x_299,
            l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_299 = l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_conv1_parameters_weight_ = (None)
        x_300 = torch.nn.functional.batch_norm(
            bottleneck_output_72,
            l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_72 = l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_norm2_parameters_bias_ = (None)
        x_301 = torch.nn.functional.relu(x_300, inplace=True)
        x_300 = None
        new_features_72 = torch.conv2d(
            x_301,
            l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_301 = l_self_modules_features_modules_denseblock3_modules_denselayer55_modules_conv2_parameters_weight_ = (None)
        concated_features_73 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
                new_features_64,
                new_features_65,
                new_features_66,
                new_features_67,
                new_features_68,
                new_features_69,
                new_features_70,
                new_features_71,
                new_features_72,
            ],
            1,
        )
        x_302 = torch.nn.functional.batch_norm(
            concated_features_73,
            l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_73 = l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm1_parameters_bias_ = (None)
        x_303 = torch.nn.functional.relu(x_302, inplace=True)
        x_302 = None
        bottleneck_output_73 = torch.conv2d(
            x_303,
            l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_303 = l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_conv1_parameters_weight_ = (None)
        x_304 = torch.nn.functional.batch_norm(
            bottleneck_output_73,
            l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_73 = l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_norm2_parameters_bias_ = (None)
        x_305 = torch.nn.functional.relu(x_304, inplace=True)
        x_304 = None
        new_features_73 = torch.conv2d(
            x_305,
            l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_305 = l_self_modules_features_modules_denseblock3_modules_denselayer56_modules_conv2_parameters_weight_ = (None)
        concated_features_74 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
                new_features_64,
                new_features_65,
                new_features_66,
                new_features_67,
                new_features_68,
                new_features_69,
                new_features_70,
                new_features_71,
                new_features_72,
                new_features_73,
            ],
            1,
        )
        x_306 = torch.nn.functional.batch_norm(
            concated_features_74,
            l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_74 = l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm1_parameters_bias_ = (None)
        x_307 = torch.nn.functional.relu(x_306, inplace=True)
        x_306 = None
        bottleneck_output_74 = torch.conv2d(
            x_307,
            l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_307 = l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_conv1_parameters_weight_ = (None)
        x_308 = torch.nn.functional.batch_norm(
            bottleneck_output_74,
            l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_74 = l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_norm2_parameters_bias_ = (None)
        x_309 = torch.nn.functional.relu(x_308, inplace=True)
        x_308 = None
        new_features_74 = torch.conv2d(
            x_309,
            l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_309 = l_self_modules_features_modules_denseblock3_modules_denselayer57_modules_conv2_parameters_weight_ = (None)
        concated_features_75 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
                new_features_64,
                new_features_65,
                new_features_66,
                new_features_67,
                new_features_68,
                new_features_69,
                new_features_70,
                new_features_71,
                new_features_72,
                new_features_73,
                new_features_74,
            ],
            1,
        )
        x_310 = torch.nn.functional.batch_norm(
            concated_features_75,
            l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_75 = l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm1_parameters_bias_ = (None)
        x_311 = torch.nn.functional.relu(x_310, inplace=True)
        x_310 = None
        bottleneck_output_75 = torch.conv2d(
            x_311,
            l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_311 = l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_conv1_parameters_weight_ = (None)
        x_312 = torch.nn.functional.batch_norm(
            bottleneck_output_75,
            l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_75 = l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_norm2_parameters_bias_ = (None)
        x_313 = torch.nn.functional.relu(x_312, inplace=True)
        x_312 = None
        new_features_75 = torch.conv2d(
            x_313,
            l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_313 = l_self_modules_features_modules_denseblock3_modules_denselayer58_modules_conv2_parameters_weight_ = (None)
        concated_features_76 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
                new_features_64,
                new_features_65,
                new_features_66,
                new_features_67,
                new_features_68,
                new_features_69,
                new_features_70,
                new_features_71,
                new_features_72,
                new_features_73,
                new_features_74,
                new_features_75,
            ],
            1,
        )
        x_314 = torch.nn.functional.batch_norm(
            concated_features_76,
            l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_76 = l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm1_parameters_bias_ = (None)
        x_315 = torch.nn.functional.relu(x_314, inplace=True)
        x_314 = None
        bottleneck_output_76 = torch.conv2d(
            x_315,
            l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_315 = l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_conv1_parameters_weight_ = (None)
        x_316 = torch.nn.functional.batch_norm(
            bottleneck_output_76,
            l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_76 = l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_norm2_parameters_bias_ = (None)
        x_317 = torch.nn.functional.relu(x_316, inplace=True)
        x_316 = None
        new_features_76 = torch.conv2d(
            x_317,
            l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_317 = l_self_modules_features_modules_denseblock3_modules_denselayer59_modules_conv2_parameters_weight_ = (None)
        concated_features_77 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
                new_features_64,
                new_features_65,
                new_features_66,
                new_features_67,
                new_features_68,
                new_features_69,
                new_features_70,
                new_features_71,
                new_features_72,
                new_features_73,
                new_features_74,
                new_features_75,
                new_features_76,
            ],
            1,
        )
        x_318 = torch.nn.functional.batch_norm(
            concated_features_77,
            l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_77 = l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm1_parameters_bias_ = (None)
        x_319 = torch.nn.functional.relu(x_318, inplace=True)
        x_318 = None
        bottleneck_output_77 = torch.conv2d(
            x_319,
            l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_319 = l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_conv1_parameters_weight_ = (None)
        x_320 = torch.nn.functional.batch_norm(
            bottleneck_output_77,
            l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_77 = l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_norm2_parameters_bias_ = (None)
        x_321 = torch.nn.functional.relu(x_320, inplace=True)
        x_320 = None
        new_features_77 = torch.conv2d(
            x_321,
            l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_321 = l_self_modules_features_modules_denseblock3_modules_denselayer60_modules_conv2_parameters_weight_ = (None)
        concated_features_78 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
                new_features_64,
                new_features_65,
                new_features_66,
                new_features_67,
                new_features_68,
                new_features_69,
                new_features_70,
                new_features_71,
                new_features_72,
                new_features_73,
                new_features_74,
                new_features_75,
                new_features_76,
                new_features_77,
            ],
            1,
        )
        x_322 = torch.nn.functional.batch_norm(
            concated_features_78,
            l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_78 = l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm1_parameters_bias_ = (None)
        x_323 = torch.nn.functional.relu(x_322, inplace=True)
        x_322 = None
        bottleneck_output_78 = torch.conv2d(
            x_323,
            l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_323 = l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_conv1_parameters_weight_ = (None)
        x_324 = torch.nn.functional.batch_norm(
            bottleneck_output_78,
            l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_78 = l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_norm2_parameters_bias_ = (None)
        x_325 = torch.nn.functional.relu(x_324, inplace=True)
        x_324 = None
        new_features_78 = torch.conv2d(
            x_325,
            l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_325 = l_self_modules_features_modules_denseblock3_modules_denselayer61_modules_conv2_parameters_weight_ = (None)
        concated_features_79 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
                new_features_64,
                new_features_65,
                new_features_66,
                new_features_67,
                new_features_68,
                new_features_69,
                new_features_70,
                new_features_71,
                new_features_72,
                new_features_73,
                new_features_74,
                new_features_75,
                new_features_76,
                new_features_77,
                new_features_78,
            ],
            1,
        )
        x_326 = torch.nn.functional.batch_norm(
            concated_features_79,
            l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_79 = l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm1_parameters_bias_ = (None)
        x_327 = torch.nn.functional.relu(x_326, inplace=True)
        x_326 = None
        bottleneck_output_79 = torch.conv2d(
            x_327,
            l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_327 = l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_conv1_parameters_weight_ = (None)
        x_328 = torch.nn.functional.batch_norm(
            bottleneck_output_79,
            l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_79 = l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_norm2_parameters_bias_ = (None)
        x_329 = torch.nn.functional.relu(x_328, inplace=True)
        x_328 = None
        new_features_79 = torch.conv2d(
            x_329,
            l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_329 = l_self_modules_features_modules_denseblock3_modules_denselayer62_modules_conv2_parameters_weight_ = (None)
        concated_features_80 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
                new_features_64,
                new_features_65,
                new_features_66,
                new_features_67,
                new_features_68,
                new_features_69,
                new_features_70,
                new_features_71,
                new_features_72,
                new_features_73,
                new_features_74,
                new_features_75,
                new_features_76,
                new_features_77,
                new_features_78,
                new_features_79,
            ],
            1,
        )
        x_330 = torch.nn.functional.batch_norm(
            concated_features_80,
            l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_80 = l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm1_parameters_bias_ = (None)
        x_331 = torch.nn.functional.relu(x_330, inplace=True)
        x_330 = None
        bottleneck_output_80 = torch.conv2d(
            x_331,
            l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_331 = l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_conv1_parameters_weight_ = (None)
        x_332 = torch.nn.functional.batch_norm(
            bottleneck_output_80,
            l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_80 = l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_norm2_parameters_bias_ = (None)
        x_333 = torch.nn.functional.relu(x_332, inplace=True)
        x_332 = None
        new_features_80 = torch.conv2d(
            x_333,
            l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_333 = l_self_modules_features_modules_denseblock3_modules_denselayer63_modules_conv2_parameters_weight_ = (None)
        concated_features_81 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
                new_features_64,
                new_features_65,
                new_features_66,
                new_features_67,
                new_features_68,
                new_features_69,
                new_features_70,
                new_features_71,
                new_features_72,
                new_features_73,
                new_features_74,
                new_features_75,
                new_features_76,
                new_features_77,
                new_features_78,
                new_features_79,
                new_features_80,
            ],
            1,
        )
        x_334 = torch.nn.functional.batch_norm(
            concated_features_81,
            l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_81 = l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm1_parameters_bias_ = (None)
        x_335 = torch.nn.functional.relu(x_334, inplace=True)
        x_334 = None
        bottleneck_output_81 = torch.conv2d(
            x_335,
            l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_335 = l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_conv1_parameters_weight_ = (None)
        x_336 = torch.nn.functional.batch_norm(
            bottleneck_output_81,
            l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_81 = l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_norm2_parameters_bias_ = (None)
        x_337 = torch.nn.functional.relu(x_336, inplace=True)
        x_336 = None
        new_features_81 = torch.conv2d(
            x_337,
            l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_337 = l_self_modules_features_modules_denseblock3_modules_denselayer64_modules_conv2_parameters_weight_ = (None)
        input_11 = torch.cat(
            [
                input_10,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
                new_features_24,
                new_features_25,
                new_features_26,
                new_features_27,
                new_features_28,
                new_features_29,
                new_features_30,
                new_features_31,
                new_features_32,
                new_features_33,
                new_features_34,
                new_features_35,
                new_features_36,
                new_features_37,
                new_features_38,
                new_features_39,
                new_features_40,
                new_features_41,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
                new_features_48,
                new_features_49,
                new_features_50,
                new_features_51,
                new_features_52,
                new_features_53,
                new_features_54,
                new_features_55,
                new_features_56,
                new_features_57,
                new_features_58,
                new_features_59,
                new_features_60,
                new_features_61,
                new_features_62,
                new_features_63,
                new_features_64,
                new_features_65,
                new_features_66,
                new_features_67,
                new_features_68,
                new_features_69,
                new_features_70,
                new_features_71,
                new_features_72,
                new_features_73,
                new_features_74,
                new_features_75,
                new_features_76,
                new_features_77,
                new_features_78,
                new_features_79,
                new_features_80,
                new_features_81,
            ],
            1,
        )
        input_10 = (
            new_features_18
        ) = (
            new_features_19
        ) = (
            new_features_20
        ) = (
            new_features_21
        ) = (
            new_features_22
        ) = (
            new_features_23
        ) = (
            new_features_24
        ) = (
            new_features_25
        ) = (
            new_features_26
        ) = (
            new_features_27
        ) = (
            new_features_28
        ) = (
            new_features_29
        ) = (
            new_features_30
        ) = (
            new_features_31
        ) = (
            new_features_32
        ) = (
            new_features_33
        ) = (
            new_features_34
        ) = (
            new_features_35
        ) = (
            new_features_36
        ) = (
            new_features_37
        ) = (
            new_features_38
        ) = (
            new_features_39
        ) = (
            new_features_40
        ) = (
            new_features_41
        ) = (
            new_features_42
        ) = (
            new_features_43
        ) = (
            new_features_44
        ) = (
            new_features_45
        ) = (
            new_features_46
        ) = (
            new_features_47
        ) = (
            new_features_48
        ) = (
            new_features_49
        ) = (
            new_features_50
        ) = (
            new_features_51
        ) = (
            new_features_52
        ) = (
            new_features_53
        ) = (
            new_features_54
        ) = (
            new_features_55
        ) = (
            new_features_56
        ) = (
            new_features_57
        ) = (
            new_features_58
        ) = (
            new_features_59
        ) = (
            new_features_60
        ) = (
            new_features_61
        ) = (
            new_features_62
        ) = (
            new_features_63
        ) = (
            new_features_64
        ) = (
            new_features_65
        ) = (
            new_features_66
        ) = (
            new_features_67
        ) = (
            new_features_68
        ) = (
            new_features_69
        ) = (
            new_features_70
        ) = (
            new_features_71
        ) = (
            new_features_72
        ) = (
            new_features_73
        ) = (
            new_features_74
        ) = (
            new_features_75
        ) = (
            new_features_76
        ) = (
            new_features_77
        ) = new_features_78 = new_features_79 = new_features_80 = new_features_81 = None
        x_338 = torch.nn.functional.batch_norm(
            input_11,
            l_self_modules_features_modules_transition3_modules_norm_buffers_running_mean_,
            l_self_modules_features_modules_transition3_modules_norm_buffers_running_var_,
            l_self_modules_features_modules_transition3_modules_norm_parameters_weight_,
            l_self_modules_features_modules_transition3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_11 = l_self_modules_features_modules_transition3_modules_norm_buffers_running_mean_ = l_self_modules_features_modules_transition3_modules_norm_buffers_running_var_ = (
            l_self_modules_features_modules_transition3_modules_norm_parameters_weight_
        ) = (
            l_self_modules_features_modules_transition3_modules_norm_parameters_bias_
        ) = None
        x_339 = torch.nn.functional.relu(x_338, inplace=True)
        x_338 = None
        input_12 = torch.conv2d(
            x_339,
            l_self_modules_features_modules_transition3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_339 = (
            l_self_modules_features_modules_transition3_modules_conv_parameters_weight_
        ) = None
        input_13 = torch._C._nn.avg_pool2d(input_12, 2, 2, 0, False, True, None)
        input_12 = None
        concated_features_82 = torch.cat([input_13], 1)
        x_340 = torch.nn.functional.batch_norm(
            concated_features_82,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_82 = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_parameters_bias_ = (None)
        x_341 = torch.nn.functional.relu(x_340, inplace=True)
        x_340 = None
        bottleneck_output_82 = torch.conv2d(
            x_341,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_341 = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_conv1_parameters_weight_ = (None)
        x_342 = torch.nn.functional.batch_norm(
            bottleneck_output_82,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_82 = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_parameters_bias_ = (None)
        x_343 = torch.nn.functional.relu(x_342, inplace=True)
        x_342 = None
        new_features_82 = torch.conv2d(
            x_343,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_343 = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_conv2_parameters_weight_ = (None)
        concated_features_83 = torch.cat([input_13, new_features_82], 1)
        x_344 = torch.nn.functional.batch_norm(
            concated_features_83,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_83 = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_parameters_bias_ = (None)
        x_345 = torch.nn.functional.relu(x_344, inplace=True)
        x_344 = None
        bottleneck_output_83 = torch.conv2d(
            x_345,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_345 = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_conv1_parameters_weight_ = (None)
        x_346 = torch.nn.functional.batch_norm(
            bottleneck_output_83,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_83 = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_parameters_bias_ = (None)
        x_347 = torch.nn.functional.relu(x_346, inplace=True)
        x_346 = None
        new_features_83 = torch.conv2d(
            x_347,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_347 = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_conv2_parameters_weight_ = (None)
        concated_features_84 = torch.cat(
            [input_13, new_features_82, new_features_83], 1
        )
        x_348 = torch.nn.functional.batch_norm(
            concated_features_84,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_84 = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_parameters_bias_ = (None)
        x_349 = torch.nn.functional.relu(x_348, inplace=True)
        x_348 = None
        bottleneck_output_84 = torch.conv2d(
            x_349,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_349 = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_conv1_parameters_weight_ = (None)
        x_350 = torch.nn.functional.batch_norm(
            bottleneck_output_84,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_84 = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_parameters_bias_ = (None)
        x_351 = torch.nn.functional.relu(x_350, inplace=True)
        x_350 = None
        new_features_84 = torch.conv2d(
            x_351,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_351 = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_conv2_parameters_weight_ = (None)
        concated_features_85 = torch.cat(
            [input_13, new_features_82, new_features_83, new_features_84], 1
        )
        x_352 = torch.nn.functional.batch_norm(
            concated_features_85,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_85 = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_parameters_bias_ = (None)
        x_353 = torch.nn.functional.relu(x_352, inplace=True)
        x_352 = None
        bottleneck_output_85 = torch.conv2d(
            x_353,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_353 = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_conv1_parameters_weight_ = (None)
        x_354 = torch.nn.functional.batch_norm(
            bottleneck_output_85,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_85 = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_parameters_bias_ = (None)
        x_355 = torch.nn.functional.relu(x_354, inplace=True)
        x_354 = None
        new_features_85 = torch.conv2d(
            x_355,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_355 = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_conv2_parameters_weight_ = (None)
        concated_features_86 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
            ],
            1,
        )
        x_356 = torch.nn.functional.batch_norm(
            concated_features_86,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_86 = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_parameters_bias_ = (None)
        x_357 = torch.nn.functional.relu(x_356, inplace=True)
        x_356 = None
        bottleneck_output_86 = torch.conv2d(
            x_357,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_357 = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_conv1_parameters_weight_ = (None)
        x_358 = torch.nn.functional.batch_norm(
            bottleneck_output_86,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_86 = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_parameters_bias_ = (None)
        x_359 = torch.nn.functional.relu(x_358, inplace=True)
        x_358 = None
        new_features_86 = torch.conv2d(
            x_359,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_359 = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_conv2_parameters_weight_ = (None)
        concated_features_87 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
            ],
            1,
        )
        x_360 = torch.nn.functional.batch_norm(
            concated_features_87,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_87 = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_parameters_bias_ = (None)
        x_361 = torch.nn.functional.relu(x_360, inplace=True)
        x_360 = None
        bottleneck_output_87 = torch.conv2d(
            x_361,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_361 = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_conv1_parameters_weight_ = (None)
        x_362 = torch.nn.functional.batch_norm(
            bottleneck_output_87,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_87 = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_parameters_bias_ = (None)
        x_363 = torch.nn.functional.relu(x_362, inplace=True)
        x_362 = None
        new_features_87 = torch.conv2d(
            x_363,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_363 = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_conv2_parameters_weight_ = (None)
        concated_features_88 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
            ],
            1,
        )
        x_364 = torch.nn.functional.batch_norm(
            concated_features_88,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_88 = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_parameters_bias_ = (None)
        x_365 = torch.nn.functional.relu(x_364, inplace=True)
        x_364 = None
        bottleneck_output_88 = torch.conv2d(
            x_365,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_365 = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_conv1_parameters_weight_ = (None)
        x_366 = torch.nn.functional.batch_norm(
            bottleneck_output_88,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_88 = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_parameters_bias_ = (None)
        x_367 = torch.nn.functional.relu(x_366, inplace=True)
        x_366 = None
        new_features_88 = torch.conv2d(
            x_367,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_367 = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_conv2_parameters_weight_ = (None)
        concated_features_89 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
            ],
            1,
        )
        x_368 = torch.nn.functional.batch_norm(
            concated_features_89,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_89 = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_parameters_bias_ = (None)
        x_369 = torch.nn.functional.relu(x_368, inplace=True)
        x_368 = None
        bottleneck_output_89 = torch.conv2d(
            x_369,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_369 = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_conv1_parameters_weight_ = (None)
        x_370 = torch.nn.functional.batch_norm(
            bottleneck_output_89,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_89 = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_parameters_bias_ = (None)
        x_371 = torch.nn.functional.relu(x_370, inplace=True)
        x_370 = None
        new_features_89 = torch.conv2d(
            x_371,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_371 = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_conv2_parameters_weight_ = (None)
        concated_features_90 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
            ],
            1,
        )
        x_372 = torch.nn.functional.batch_norm(
            concated_features_90,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_90 = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_parameters_bias_ = (None)
        x_373 = torch.nn.functional.relu(x_372, inplace=True)
        x_372 = None
        bottleneck_output_90 = torch.conv2d(
            x_373,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_373 = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_conv1_parameters_weight_ = (None)
        x_374 = torch.nn.functional.batch_norm(
            bottleneck_output_90,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_90 = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_parameters_bias_ = (None)
        x_375 = torch.nn.functional.relu(x_374, inplace=True)
        x_374 = None
        new_features_90 = torch.conv2d(
            x_375,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_375 = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_conv2_parameters_weight_ = (None)
        concated_features_91 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
            ],
            1,
        )
        x_376 = torch.nn.functional.batch_norm(
            concated_features_91,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_91 = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_parameters_bias_ = (None)
        x_377 = torch.nn.functional.relu(x_376, inplace=True)
        x_376 = None
        bottleneck_output_91 = torch.conv2d(
            x_377,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_377 = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_conv1_parameters_weight_ = (None)
        x_378 = torch.nn.functional.batch_norm(
            bottleneck_output_91,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_91 = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_parameters_bias_ = (None)
        x_379 = torch.nn.functional.relu(x_378, inplace=True)
        x_378 = None
        new_features_91 = torch.conv2d(
            x_379,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_379 = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_conv2_parameters_weight_ = (None)
        concated_features_92 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
            ],
            1,
        )
        x_380 = torch.nn.functional.batch_norm(
            concated_features_92,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_92 = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_parameters_bias_ = (None)
        x_381 = torch.nn.functional.relu(x_380, inplace=True)
        x_380 = None
        bottleneck_output_92 = torch.conv2d(
            x_381,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_381 = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_conv1_parameters_weight_ = (None)
        x_382 = torch.nn.functional.batch_norm(
            bottleneck_output_92,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_92 = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_parameters_bias_ = (None)
        x_383 = torch.nn.functional.relu(x_382, inplace=True)
        x_382 = None
        new_features_92 = torch.conv2d(
            x_383,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_383 = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_conv2_parameters_weight_ = (None)
        concated_features_93 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
            ],
            1,
        )
        x_384 = torch.nn.functional.batch_norm(
            concated_features_93,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_93 = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_parameters_bias_ = (None)
        x_385 = torch.nn.functional.relu(x_384, inplace=True)
        x_384 = None
        bottleneck_output_93 = torch.conv2d(
            x_385,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_385 = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_conv1_parameters_weight_ = (None)
        x_386 = torch.nn.functional.batch_norm(
            bottleneck_output_93,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_93 = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_parameters_bias_ = (None)
        x_387 = torch.nn.functional.relu(x_386, inplace=True)
        x_386 = None
        new_features_93 = torch.conv2d(
            x_387,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_387 = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_conv2_parameters_weight_ = (None)
        concated_features_94 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
            ],
            1,
        )
        x_388 = torch.nn.functional.batch_norm(
            concated_features_94,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_94 = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_parameters_bias_ = (None)
        x_389 = torch.nn.functional.relu(x_388, inplace=True)
        x_388 = None
        bottleneck_output_94 = torch.conv2d(
            x_389,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_389 = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_conv1_parameters_weight_ = (None)
        x_390 = torch.nn.functional.batch_norm(
            bottleneck_output_94,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_94 = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_parameters_bias_ = (None)
        x_391 = torch.nn.functional.relu(x_390, inplace=True)
        x_390 = None
        new_features_94 = torch.conv2d(
            x_391,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_391 = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_conv2_parameters_weight_ = (None)
        concated_features_95 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
            ],
            1,
        )
        x_392 = torch.nn.functional.batch_norm(
            concated_features_95,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_95 = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_parameters_bias_ = (None)
        x_393 = torch.nn.functional.relu(x_392, inplace=True)
        x_392 = None
        bottleneck_output_95 = torch.conv2d(
            x_393,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_393 = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_conv1_parameters_weight_ = (None)
        x_394 = torch.nn.functional.batch_norm(
            bottleneck_output_95,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_95 = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_parameters_bias_ = (None)
        x_395 = torch.nn.functional.relu(x_394, inplace=True)
        x_394 = None
        new_features_95 = torch.conv2d(
            x_395,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_395 = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_conv2_parameters_weight_ = (None)
        concated_features_96 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
            ],
            1,
        )
        x_396 = torch.nn.functional.batch_norm(
            concated_features_96,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_96 = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_parameters_bias_ = (None)
        x_397 = torch.nn.functional.relu(x_396, inplace=True)
        x_396 = None
        bottleneck_output_96 = torch.conv2d(
            x_397,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_397 = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_conv1_parameters_weight_ = (None)
        x_398 = torch.nn.functional.batch_norm(
            bottleneck_output_96,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_96 = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_parameters_bias_ = (None)
        x_399 = torch.nn.functional.relu(x_398, inplace=True)
        x_398 = None
        new_features_96 = torch.conv2d(
            x_399,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_399 = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_conv2_parameters_weight_ = (None)
        concated_features_97 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
            ],
            1,
        )
        x_400 = torch.nn.functional.batch_norm(
            concated_features_97,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_97 = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_parameters_bias_ = (None)
        x_401 = torch.nn.functional.relu(x_400, inplace=True)
        x_400 = None
        bottleneck_output_97 = torch.conv2d(
            x_401,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_401 = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_conv1_parameters_weight_ = (None)
        x_402 = torch.nn.functional.batch_norm(
            bottleneck_output_97,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_97 = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_parameters_bias_ = (None)
        x_403 = torch.nn.functional.relu(x_402, inplace=True)
        x_402 = None
        new_features_97 = torch.conv2d(
            x_403,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_403 = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_conv2_parameters_weight_ = (None)
        concated_features_98 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
            ],
            1,
        )
        x_404 = torch.nn.functional.batch_norm(
            concated_features_98,
            l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_98 = l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm1_parameters_bias_ = (None)
        x_405 = torch.nn.functional.relu(x_404, inplace=True)
        x_404 = None
        bottleneck_output_98 = torch.conv2d(
            x_405,
            l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_405 = l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_conv1_parameters_weight_ = (None)
        x_406 = torch.nn.functional.batch_norm(
            bottleneck_output_98,
            l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_98 = l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_norm2_parameters_bias_ = (None)
        x_407 = torch.nn.functional.relu(x_406, inplace=True)
        x_406 = None
        new_features_98 = torch.conv2d(
            x_407,
            l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_407 = l_self_modules_features_modules_denseblock4_modules_denselayer17_modules_conv2_parameters_weight_ = (None)
        concated_features_99 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
            ],
            1,
        )
        x_408 = torch.nn.functional.batch_norm(
            concated_features_99,
            l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_99 = l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm1_parameters_bias_ = (None)
        x_409 = torch.nn.functional.relu(x_408, inplace=True)
        x_408 = None
        bottleneck_output_99 = torch.conv2d(
            x_409,
            l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_409 = l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_conv1_parameters_weight_ = (None)
        x_410 = torch.nn.functional.batch_norm(
            bottleneck_output_99,
            l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_99 = l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_norm2_parameters_bias_ = (None)
        x_411 = torch.nn.functional.relu(x_410, inplace=True)
        x_410 = None
        new_features_99 = torch.conv2d(
            x_411,
            l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_411 = l_self_modules_features_modules_denseblock4_modules_denselayer18_modules_conv2_parameters_weight_ = (None)
        concated_features_100 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
            ],
            1,
        )
        x_412 = torch.nn.functional.batch_norm(
            concated_features_100,
            l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_100 = l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm1_parameters_bias_ = (None)
        x_413 = torch.nn.functional.relu(x_412, inplace=True)
        x_412 = None
        bottleneck_output_100 = torch.conv2d(
            x_413,
            l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_413 = l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_conv1_parameters_weight_ = (None)
        x_414 = torch.nn.functional.batch_norm(
            bottleneck_output_100,
            l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_100 = l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_norm2_parameters_bias_ = (None)
        x_415 = torch.nn.functional.relu(x_414, inplace=True)
        x_414 = None
        new_features_100 = torch.conv2d(
            x_415,
            l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_415 = l_self_modules_features_modules_denseblock4_modules_denselayer19_modules_conv2_parameters_weight_ = (None)
        concated_features_101 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
            ],
            1,
        )
        x_416 = torch.nn.functional.batch_norm(
            concated_features_101,
            l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_101 = l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm1_parameters_bias_ = (None)
        x_417 = torch.nn.functional.relu(x_416, inplace=True)
        x_416 = None
        bottleneck_output_101 = torch.conv2d(
            x_417,
            l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_417 = l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_conv1_parameters_weight_ = (None)
        x_418 = torch.nn.functional.batch_norm(
            bottleneck_output_101,
            l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_101 = l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_norm2_parameters_bias_ = (None)
        x_419 = torch.nn.functional.relu(x_418, inplace=True)
        x_418 = None
        new_features_101 = torch.conv2d(
            x_419,
            l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_419 = l_self_modules_features_modules_denseblock4_modules_denselayer20_modules_conv2_parameters_weight_ = (None)
        concated_features_102 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
            ],
            1,
        )
        x_420 = torch.nn.functional.batch_norm(
            concated_features_102,
            l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_102 = l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm1_parameters_bias_ = (None)
        x_421 = torch.nn.functional.relu(x_420, inplace=True)
        x_420 = None
        bottleneck_output_102 = torch.conv2d(
            x_421,
            l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_421 = l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_conv1_parameters_weight_ = (None)
        x_422 = torch.nn.functional.batch_norm(
            bottleneck_output_102,
            l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_102 = l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_norm2_parameters_bias_ = (None)
        x_423 = torch.nn.functional.relu(x_422, inplace=True)
        x_422 = None
        new_features_102 = torch.conv2d(
            x_423,
            l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_423 = l_self_modules_features_modules_denseblock4_modules_denselayer21_modules_conv2_parameters_weight_ = (None)
        concated_features_103 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
            ],
            1,
        )
        x_424 = torch.nn.functional.batch_norm(
            concated_features_103,
            l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_103 = l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm1_parameters_bias_ = (None)
        x_425 = torch.nn.functional.relu(x_424, inplace=True)
        x_424 = None
        bottleneck_output_103 = torch.conv2d(
            x_425,
            l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_425 = l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_conv1_parameters_weight_ = (None)
        x_426 = torch.nn.functional.batch_norm(
            bottleneck_output_103,
            l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_103 = l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_norm2_parameters_bias_ = (None)
        x_427 = torch.nn.functional.relu(x_426, inplace=True)
        x_426 = None
        new_features_103 = torch.conv2d(
            x_427,
            l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_427 = l_self_modules_features_modules_denseblock4_modules_denselayer22_modules_conv2_parameters_weight_ = (None)
        concated_features_104 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
            ],
            1,
        )
        x_428 = torch.nn.functional.batch_norm(
            concated_features_104,
            l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_104 = l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm1_parameters_bias_ = (None)
        x_429 = torch.nn.functional.relu(x_428, inplace=True)
        x_428 = None
        bottleneck_output_104 = torch.conv2d(
            x_429,
            l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_429 = l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_conv1_parameters_weight_ = (None)
        x_430 = torch.nn.functional.batch_norm(
            bottleneck_output_104,
            l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_104 = l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_norm2_parameters_bias_ = (None)
        x_431 = torch.nn.functional.relu(x_430, inplace=True)
        x_430 = None
        new_features_104 = torch.conv2d(
            x_431,
            l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_431 = l_self_modules_features_modules_denseblock4_modules_denselayer23_modules_conv2_parameters_weight_ = (None)
        concated_features_105 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
            ],
            1,
        )
        x_432 = torch.nn.functional.batch_norm(
            concated_features_105,
            l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_105 = l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm1_parameters_bias_ = (None)
        x_433 = torch.nn.functional.relu(x_432, inplace=True)
        x_432 = None
        bottleneck_output_105 = torch.conv2d(
            x_433,
            l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_433 = l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_conv1_parameters_weight_ = (None)
        x_434 = torch.nn.functional.batch_norm(
            bottleneck_output_105,
            l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_105 = l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_norm2_parameters_bias_ = (None)
        x_435 = torch.nn.functional.relu(x_434, inplace=True)
        x_434 = None
        new_features_105 = torch.conv2d(
            x_435,
            l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_435 = l_self_modules_features_modules_denseblock4_modules_denselayer24_modules_conv2_parameters_weight_ = (None)
        concated_features_106 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
            ],
            1,
        )
        x_436 = torch.nn.functional.batch_norm(
            concated_features_106,
            l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_106 = l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm1_parameters_bias_ = (None)
        x_437 = torch.nn.functional.relu(x_436, inplace=True)
        x_436 = None
        bottleneck_output_106 = torch.conv2d(
            x_437,
            l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_437 = l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_conv1_parameters_weight_ = (None)
        x_438 = torch.nn.functional.batch_norm(
            bottleneck_output_106,
            l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_106 = l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_norm2_parameters_bias_ = (None)
        x_439 = torch.nn.functional.relu(x_438, inplace=True)
        x_438 = None
        new_features_106 = torch.conv2d(
            x_439,
            l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_439 = l_self_modules_features_modules_denseblock4_modules_denselayer25_modules_conv2_parameters_weight_ = (None)
        concated_features_107 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
            ],
            1,
        )
        x_440 = torch.nn.functional.batch_norm(
            concated_features_107,
            l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_107 = l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm1_parameters_bias_ = (None)
        x_441 = torch.nn.functional.relu(x_440, inplace=True)
        x_440 = None
        bottleneck_output_107 = torch.conv2d(
            x_441,
            l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_441 = l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_conv1_parameters_weight_ = (None)
        x_442 = torch.nn.functional.batch_norm(
            bottleneck_output_107,
            l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_107 = l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_norm2_parameters_bias_ = (None)
        x_443 = torch.nn.functional.relu(x_442, inplace=True)
        x_442 = None
        new_features_107 = torch.conv2d(
            x_443,
            l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_443 = l_self_modules_features_modules_denseblock4_modules_denselayer26_modules_conv2_parameters_weight_ = (None)
        concated_features_108 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
            ],
            1,
        )
        x_444 = torch.nn.functional.batch_norm(
            concated_features_108,
            l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_108 = l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm1_parameters_bias_ = (None)
        x_445 = torch.nn.functional.relu(x_444, inplace=True)
        x_444 = None
        bottleneck_output_108 = torch.conv2d(
            x_445,
            l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_445 = l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_conv1_parameters_weight_ = (None)
        x_446 = torch.nn.functional.batch_norm(
            bottleneck_output_108,
            l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_108 = l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_norm2_parameters_bias_ = (None)
        x_447 = torch.nn.functional.relu(x_446, inplace=True)
        x_446 = None
        new_features_108 = torch.conv2d(
            x_447,
            l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_447 = l_self_modules_features_modules_denseblock4_modules_denselayer27_modules_conv2_parameters_weight_ = (None)
        concated_features_109 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
            ],
            1,
        )
        x_448 = torch.nn.functional.batch_norm(
            concated_features_109,
            l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_109 = l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm1_parameters_bias_ = (None)
        x_449 = torch.nn.functional.relu(x_448, inplace=True)
        x_448 = None
        bottleneck_output_109 = torch.conv2d(
            x_449,
            l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_449 = l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_conv1_parameters_weight_ = (None)
        x_450 = torch.nn.functional.batch_norm(
            bottleneck_output_109,
            l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_109 = l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_norm2_parameters_bias_ = (None)
        x_451 = torch.nn.functional.relu(x_450, inplace=True)
        x_450 = None
        new_features_109 = torch.conv2d(
            x_451,
            l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_451 = l_self_modules_features_modules_denseblock4_modules_denselayer28_modules_conv2_parameters_weight_ = (None)
        concated_features_110 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
            ],
            1,
        )
        x_452 = torch.nn.functional.batch_norm(
            concated_features_110,
            l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_110 = l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm1_parameters_bias_ = (None)
        x_453 = torch.nn.functional.relu(x_452, inplace=True)
        x_452 = None
        bottleneck_output_110 = torch.conv2d(
            x_453,
            l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_453 = l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_conv1_parameters_weight_ = (None)
        x_454 = torch.nn.functional.batch_norm(
            bottleneck_output_110,
            l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_110 = l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_norm2_parameters_bias_ = (None)
        x_455 = torch.nn.functional.relu(x_454, inplace=True)
        x_454 = None
        new_features_110 = torch.conv2d(
            x_455,
            l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_455 = l_self_modules_features_modules_denseblock4_modules_denselayer29_modules_conv2_parameters_weight_ = (None)
        concated_features_111 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
            ],
            1,
        )
        x_456 = torch.nn.functional.batch_norm(
            concated_features_111,
            l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_111 = l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm1_parameters_bias_ = (None)
        x_457 = torch.nn.functional.relu(x_456, inplace=True)
        x_456 = None
        bottleneck_output_111 = torch.conv2d(
            x_457,
            l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_457 = l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_conv1_parameters_weight_ = (None)
        x_458 = torch.nn.functional.batch_norm(
            bottleneck_output_111,
            l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_111 = l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_norm2_parameters_bias_ = (None)
        x_459 = torch.nn.functional.relu(x_458, inplace=True)
        x_458 = None
        new_features_111 = torch.conv2d(
            x_459,
            l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_459 = l_self_modules_features_modules_denseblock4_modules_denselayer30_modules_conv2_parameters_weight_ = (None)
        concated_features_112 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
            ],
            1,
        )
        x_460 = torch.nn.functional.batch_norm(
            concated_features_112,
            l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_112 = l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm1_parameters_bias_ = (None)
        x_461 = torch.nn.functional.relu(x_460, inplace=True)
        x_460 = None
        bottleneck_output_112 = torch.conv2d(
            x_461,
            l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_461 = l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_conv1_parameters_weight_ = (None)
        x_462 = torch.nn.functional.batch_norm(
            bottleneck_output_112,
            l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_112 = l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_norm2_parameters_bias_ = (None)
        x_463 = torch.nn.functional.relu(x_462, inplace=True)
        x_462 = None
        new_features_112 = torch.conv2d(
            x_463,
            l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_463 = l_self_modules_features_modules_denseblock4_modules_denselayer31_modules_conv2_parameters_weight_ = (None)
        concated_features_113 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
                new_features_112,
            ],
            1,
        )
        x_464 = torch.nn.functional.batch_norm(
            concated_features_113,
            l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_113 = l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm1_parameters_bias_ = (None)
        x_465 = torch.nn.functional.relu(x_464, inplace=True)
        x_464 = None
        bottleneck_output_113 = torch.conv2d(
            x_465,
            l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_465 = l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_conv1_parameters_weight_ = (None)
        x_466 = torch.nn.functional.batch_norm(
            bottleneck_output_113,
            l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_113 = l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_norm2_parameters_bias_ = (None)
        x_467 = torch.nn.functional.relu(x_466, inplace=True)
        x_466 = None
        new_features_113 = torch.conv2d(
            x_467,
            l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_467 = l_self_modules_features_modules_denseblock4_modules_denselayer32_modules_conv2_parameters_weight_ = (None)
        concated_features_114 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
                new_features_112,
                new_features_113,
            ],
            1,
        )
        x_468 = torch.nn.functional.batch_norm(
            concated_features_114,
            l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_114 = l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm1_parameters_bias_ = (None)
        x_469 = torch.nn.functional.relu(x_468, inplace=True)
        x_468 = None
        bottleneck_output_114 = torch.conv2d(
            x_469,
            l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_469 = l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_conv1_parameters_weight_ = (None)
        x_470 = torch.nn.functional.batch_norm(
            bottleneck_output_114,
            l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_114 = l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_norm2_parameters_bias_ = (None)
        x_471 = torch.nn.functional.relu(x_470, inplace=True)
        x_470 = None
        new_features_114 = torch.conv2d(
            x_471,
            l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_471 = l_self_modules_features_modules_denseblock4_modules_denselayer33_modules_conv2_parameters_weight_ = (None)
        concated_features_115 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
                new_features_112,
                new_features_113,
                new_features_114,
            ],
            1,
        )
        x_472 = torch.nn.functional.batch_norm(
            concated_features_115,
            l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_115 = l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm1_parameters_bias_ = (None)
        x_473 = torch.nn.functional.relu(x_472, inplace=True)
        x_472 = None
        bottleneck_output_115 = torch.conv2d(
            x_473,
            l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_473 = l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_conv1_parameters_weight_ = (None)
        x_474 = torch.nn.functional.batch_norm(
            bottleneck_output_115,
            l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_115 = l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_norm2_parameters_bias_ = (None)
        x_475 = torch.nn.functional.relu(x_474, inplace=True)
        x_474 = None
        new_features_115 = torch.conv2d(
            x_475,
            l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_475 = l_self_modules_features_modules_denseblock4_modules_denselayer34_modules_conv2_parameters_weight_ = (None)
        concated_features_116 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
                new_features_112,
                new_features_113,
                new_features_114,
                new_features_115,
            ],
            1,
        )
        x_476 = torch.nn.functional.batch_norm(
            concated_features_116,
            l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_116 = l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm1_parameters_bias_ = (None)
        x_477 = torch.nn.functional.relu(x_476, inplace=True)
        x_476 = None
        bottleneck_output_116 = torch.conv2d(
            x_477,
            l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_477 = l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_conv1_parameters_weight_ = (None)
        x_478 = torch.nn.functional.batch_norm(
            bottleneck_output_116,
            l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_116 = l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_norm2_parameters_bias_ = (None)
        x_479 = torch.nn.functional.relu(x_478, inplace=True)
        x_478 = None
        new_features_116 = torch.conv2d(
            x_479,
            l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_479 = l_self_modules_features_modules_denseblock4_modules_denselayer35_modules_conv2_parameters_weight_ = (None)
        concated_features_117 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
                new_features_112,
                new_features_113,
                new_features_114,
                new_features_115,
                new_features_116,
            ],
            1,
        )
        x_480 = torch.nn.functional.batch_norm(
            concated_features_117,
            l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_117 = l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm1_parameters_bias_ = (None)
        x_481 = torch.nn.functional.relu(x_480, inplace=True)
        x_480 = None
        bottleneck_output_117 = torch.conv2d(
            x_481,
            l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_481 = l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_conv1_parameters_weight_ = (None)
        x_482 = torch.nn.functional.batch_norm(
            bottleneck_output_117,
            l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_117 = l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_norm2_parameters_bias_ = (None)
        x_483 = torch.nn.functional.relu(x_482, inplace=True)
        x_482 = None
        new_features_117 = torch.conv2d(
            x_483,
            l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_483 = l_self_modules_features_modules_denseblock4_modules_denselayer36_modules_conv2_parameters_weight_ = (None)
        concated_features_118 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
                new_features_112,
                new_features_113,
                new_features_114,
                new_features_115,
                new_features_116,
                new_features_117,
            ],
            1,
        )
        x_484 = torch.nn.functional.batch_norm(
            concated_features_118,
            l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_118 = l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm1_parameters_bias_ = (None)
        x_485 = torch.nn.functional.relu(x_484, inplace=True)
        x_484 = None
        bottleneck_output_118 = torch.conv2d(
            x_485,
            l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_485 = l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_conv1_parameters_weight_ = (None)
        x_486 = torch.nn.functional.batch_norm(
            bottleneck_output_118,
            l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_118 = l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_norm2_parameters_bias_ = (None)
        x_487 = torch.nn.functional.relu(x_486, inplace=True)
        x_486 = None
        new_features_118 = torch.conv2d(
            x_487,
            l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_487 = l_self_modules_features_modules_denseblock4_modules_denselayer37_modules_conv2_parameters_weight_ = (None)
        concated_features_119 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
                new_features_112,
                new_features_113,
                new_features_114,
                new_features_115,
                new_features_116,
                new_features_117,
                new_features_118,
            ],
            1,
        )
        x_488 = torch.nn.functional.batch_norm(
            concated_features_119,
            l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_119 = l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm1_parameters_bias_ = (None)
        x_489 = torch.nn.functional.relu(x_488, inplace=True)
        x_488 = None
        bottleneck_output_119 = torch.conv2d(
            x_489,
            l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_489 = l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_conv1_parameters_weight_ = (None)
        x_490 = torch.nn.functional.batch_norm(
            bottleneck_output_119,
            l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_119 = l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_norm2_parameters_bias_ = (None)
        x_491 = torch.nn.functional.relu(x_490, inplace=True)
        x_490 = None
        new_features_119 = torch.conv2d(
            x_491,
            l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_491 = l_self_modules_features_modules_denseblock4_modules_denselayer38_modules_conv2_parameters_weight_ = (None)
        concated_features_120 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
                new_features_112,
                new_features_113,
                new_features_114,
                new_features_115,
                new_features_116,
                new_features_117,
                new_features_118,
                new_features_119,
            ],
            1,
        )
        x_492 = torch.nn.functional.batch_norm(
            concated_features_120,
            l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_120 = l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm1_parameters_bias_ = (None)
        x_493 = torch.nn.functional.relu(x_492, inplace=True)
        x_492 = None
        bottleneck_output_120 = torch.conv2d(
            x_493,
            l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_493 = l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_conv1_parameters_weight_ = (None)
        x_494 = torch.nn.functional.batch_norm(
            bottleneck_output_120,
            l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_120 = l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_norm2_parameters_bias_ = (None)
        x_495 = torch.nn.functional.relu(x_494, inplace=True)
        x_494 = None
        new_features_120 = torch.conv2d(
            x_495,
            l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_495 = l_self_modules_features_modules_denseblock4_modules_denselayer39_modules_conv2_parameters_weight_ = (None)
        concated_features_121 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
                new_features_112,
                new_features_113,
                new_features_114,
                new_features_115,
                new_features_116,
                new_features_117,
                new_features_118,
                new_features_119,
                new_features_120,
            ],
            1,
        )
        x_496 = torch.nn.functional.batch_norm(
            concated_features_121,
            l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_121 = l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm1_parameters_bias_ = (None)
        x_497 = torch.nn.functional.relu(x_496, inplace=True)
        x_496 = None
        bottleneck_output_121 = torch.conv2d(
            x_497,
            l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_497 = l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_conv1_parameters_weight_ = (None)
        x_498 = torch.nn.functional.batch_norm(
            bottleneck_output_121,
            l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_121 = l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_norm2_parameters_bias_ = (None)
        x_499 = torch.nn.functional.relu(x_498, inplace=True)
        x_498 = None
        new_features_121 = torch.conv2d(
            x_499,
            l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_499 = l_self_modules_features_modules_denseblock4_modules_denselayer40_modules_conv2_parameters_weight_ = (None)
        concated_features_122 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
                new_features_112,
                new_features_113,
                new_features_114,
                new_features_115,
                new_features_116,
                new_features_117,
                new_features_118,
                new_features_119,
                new_features_120,
                new_features_121,
            ],
            1,
        )
        x_500 = torch.nn.functional.batch_norm(
            concated_features_122,
            l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_122 = l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm1_parameters_bias_ = (None)
        x_501 = torch.nn.functional.relu(x_500, inplace=True)
        x_500 = None
        bottleneck_output_122 = torch.conv2d(
            x_501,
            l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_501 = l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_conv1_parameters_weight_ = (None)
        x_502 = torch.nn.functional.batch_norm(
            bottleneck_output_122,
            l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_122 = l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_norm2_parameters_bias_ = (None)
        x_503 = torch.nn.functional.relu(x_502, inplace=True)
        x_502 = None
        new_features_122 = torch.conv2d(
            x_503,
            l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_503 = l_self_modules_features_modules_denseblock4_modules_denselayer41_modules_conv2_parameters_weight_ = (None)
        concated_features_123 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
                new_features_112,
                new_features_113,
                new_features_114,
                new_features_115,
                new_features_116,
                new_features_117,
                new_features_118,
                new_features_119,
                new_features_120,
                new_features_121,
                new_features_122,
            ],
            1,
        )
        x_504 = torch.nn.functional.batch_norm(
            concated_features_123,
            l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_123 = l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm1_parameters_bias_ = (None)
        x_505 = torch.nn.functional.relu(x_504, inplace=True)
        x_504 = None
        bottleneck_output_123 = torch.conv2d(
            x_505,
            l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_505 = l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_conv1_parameters_weight_ = (None)
        x_506 = torch.nn.functional.batch_norm(
            bottleneck_output_123,
            l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_123 = l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_norm2_parameters_bias_ = (None)
        x_507 = torch.nn.functional.relu(x_506, inplace=True)
        x_506 = None
        new_features_123 = torch.conv2d(
            x_507,
            l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_507 = l_self_modules_features_modules_denseblock4_modules_denselayer42_modules_conv2_parameters_weight_ = (None)
        concated_features_124 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
                new_features_112,
                new_features_113,
                new_features_114,
                new_features_115,
                new_features_116,
                new_features_117,
                new_features_118,
                new_features_119,
                new_features_120,
                new_features_121,
                new_features_122,
                new_features_123,
            ],
            1,
        )
        x_508 = torch.nn.functional.batch_norm(
            concated_features_124,
            l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_124 = l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm1_parameters_bias_ = (None)
        x_509 = torch.nn.functional.relu(x_508, inplace=True)
        x_508 = None
        bottleneck_output_124 = torch.conv2d(
            x_509,
            l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_509 = l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_conv1_parameters_weight_ = (None)
        x_510 = torch.nn.functional.batch_norm(
            bottleneck_output_124,
            l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_124 = l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_norm2_parameters_bias_ = (None)
        x_511 = torch.nn.functional.relu(x_510, inplace=True)
        x_510 = None
        new_features_124 = torch.conv2d(
            x_511,
            l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_511 = l_self_modules_features_modules_denseblock4_modules_denselayer43_modules_conv2_parameters_weight_ = (None)
        concated_features_125 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
                new_features_112,
                new_features_113,
                new_features_114,
                new_features_115,
                new_features_116,
                new_features_117,
                new_features_118,
                new_features_119,
                new_features_120,
                new_features_121,
                new_features_122,
                new_features_123,
                new_features_124,
            ],
            1,
        )
        x_512 = torch.nn.functional.batch_norm(
            concated_features_125,
            l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_125 = l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm1_parameters_bias_ = (None)
        x_513 = torch.nn.functional.relu(x_512, inplace=True)
        x_512 = None
        bottleneck_output_125 = torch.conv2d(
            x_513,
            l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_513 = l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_conv1_parameters_weight_ = (None)
        x_514 = torch.nn.functional.batch_norm(
            bottleneck_output_125,
            l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_125 = l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_norm2_parameters_bias_ = (None)
        x_515 = torch.nn.functional.relu(x_514, inplace=True)
        x_514 = None
        new_features_125 = torch.conv2d(
            x_515,
            l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_515 = l_self_modules_features_modules_denseblock4_modules_denselayer44_modules_conv2_parameters_weight_ = (None)
        concated_features_126 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
                new_features_112,
                new_features_113,
                new_features_114,
                new_features_115,
                new_features_116,
                new_features_117,
                new_features_118,
                new_features_119,
                new_features_120,
                new_features_121,
                new_features_122,
                new_features_123,
                new_features_124,
                new_features_125,
            ],
            1,
        )
        x_516 = torch.nn.functional.batch_norm(
            concated_features_126,
            l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_126 = l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm1_parameters_bias_ = (None)
        x_517 = torch.nn.functional.relu(x_516, inplace=True)
        x_516 = None
        bottleneck_output_126 = torch.conv2d(
            x_517,
            l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_517 = l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_conv1_parameters_weight_ = (None)
        x_518 = torch.nn.functional.batch_norm(
            bottleneck_output_126,
            l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_126 = l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_norm2_parameters_bias_ = (None)
        x_519 = torch.nn.functional.relu(x_518, inplace=True)
        x_518 = None
        new_features_126 = torch.conv2d(
            x_519,
            l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_519 = l_self_modules_features_modules_denseblock4_modules_denselayer45_modules_conv2_parameters_weight_ = (None)
        concated_features_127 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
                new_features_112,
                new_features_113,
                new_features_114,
                new_features_115,
                new_features_116,
                new_features_117,
                new_features_118,
                new_features_119,
                new_features_120,
                new_features_121,
                new_features_122,
                new_features_123,
                new_features_124,
                new_features_125,
                new_features_126,
            ],
            1,
        )
        x_520 = torch.nn.functional.batch_norm(
            concated_features_127,
            l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_127 = l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm1_parameters_bias_ = (None)
        x_521 = torch.nn.functional.relu(x_520, inplace=True)
        x_520 = None
        bottleneck_output_127 = torch.conv2d(
            x_521,
            l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_521 = l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_conv1_parameters_weight_ = (None)
        x_522 = torch.nn.functional.batch_norm(
            bottleneck_output_127,
            l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_127 = l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_norm2_parameters_bias_ = (None)
        x_523 = torch.nn.functional.relu(x_522, inplace=True)
        x_522 = None
        new_features_127 = torch.conv2d(
            x_523,
            l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_523 = l_self_modules_features_modules_denseblock4_modules_denselayer46_modules_conv2_parameters_weight_ = (None)
        concated_features_128 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
                new_features_112,
                new_features_113,
                new_features_114,
                new_features_115,
                new_features_116,
                new_features_117,
                new_features_118,
                new_features_119,
                new_features_120,
                new_features_121,
                new_features_122,
                new_features_123,
                new_features_124,
                new_features_125,
                new_features_126,
                new_features_127,
            ],
            1,
        )
        x_524 = torch.nn.functional.batch_norm(
            concated_features_128,
            l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_128 = l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm1_parameters_bias_ = (None)
        x_525 = torch.nn.functional.relu(x_524, inplace=True)
        x_524 = None
        bottleneck_output_128 = torch.conv2d(
            x_525,
            l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_525 = l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_conv1_parameters_weight_ = (None)
        x_526 = torch.nn.functional.batch_norm(
            bottleneck_output_128,
            l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_128 = l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_norm2_parameters_bias_ = (None)
        x_527 = torch.nn.functional.relu(x_526, inplace=True)
        x_526 = None
        new_features_128 = torch.conv2d(
            x_527,
            l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_527 = l_self_modules_features_modules_denseblock4_modules_denselayer47_modules_conv2_parameters_weight_ = (None)
        concated_features_129 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
                new_features_112,
                new_features_113,
                new_features_114,
                new_features_115,
                new_features_116,
                new_features_117,
                new_features_118,
                new_features_119,
                new_features_120,
                new_features_121,
                new_features_122,
                new_features_123,
                new_features_124,
                new_features_125,
                new_features_126,
                new_features_127,
                new_features_128,
            ],
            1,
        )
        x_528 = torch.nn.functional.batch_norm(
            concated_features_129,
            l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_129 = l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm1_parameters_bias_ = (None)
        x_529 = torch.nn.functional.relu(x_528, inplace=True)
        x_528 = None
        bottleneck_output_129 = torch.conv2d(
            x_529,
            l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_529 = l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_conv1_parameters_weight_ = (None)
        x_530 = torch.nn.functional.batch_norm(
            bottleneck_output_129,
            l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_129 = l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_norm2_parameters_bias_ = (None)
        x_531 = torch.nn.functional.relu(x_530, inplace=True)
        x_530 = None
        new_features_129 = torch.conv2d(
            x_531,
            l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_531 = l_self_modules_features_modules_denseblock4_modules_denselayer48_modules_conv2_parameters_weight_ = (None)
        input_14 = torch.cat(
            [
                input_13,
                new_features_82,
                new_features_83,
                new_features_84,
                new_features_85,
                new_features_86,
                new_features_87,
                new_features_88,
                new_features_89,
                new_features_90,
                new_features_91,
                new_features_92,
                new_features_93,
                new_features_94,
                new_features_95,
                new_features_96,
                new_features_97,
                new_features_98,
                new_features_99,
                new_features_100,
                new_features_101,
                new_features_102,
                new_features_103,
                new_features_104,
                new_features_105,
                new_features_106,
                new_features_107,
                new_features_108,
                new_features_109,
                new_features_110,
                new_features_111,
                new_features_112,
                new_features_113,
                new_features_114,
                new_features_115,
                new_features_116,
                new_features_117,
                new_features_118,
                new_features_119,
                new_features_120,
                new_features_121,
                new_features_122,
                new_features_123,
                new_features_124,
                new_features_125,
                new_features_126,
                new_features_127,
                new_features_128,
                new_features_129,
            ],
            1,
        )
        input_13 = (
            new_features_82
        ) = (
            new_features_83
        ) = (
            new_features_84
        ) = (
            new_features_85
        ) = (
            new_features_86
        ) = (
            new_features_87
        ) = (
            new_features_88
        ) = (
            new_features_89
        ) = (
            new_features_90
        ) = (
            new_features_91
        ) = (
            new_features_92
        ) = (
            new_features_93
        ) = (
            new_features_94
        ) = (
            new_features_95
        ) = (
            new_features_96
        ) = (
            new_features_97
        ) = (
            new_features_98
        ) = (
            new_features_99
        ) = (
            new_features_100
        ) = (
            new_features_101
        ) = (
            new_features_102
        ) = (
            new_features_103
        ) = (
            new_features_104
        ) = (
            new_features_105
        ) = (
            new_features_106
        ) = (
            new_features_107
        ) = (
            new_features_108
        ) = (
            new_features_109
        ) = (
            new_features_110
        ) = (
            new_features_111
        ) = (
            new_features_112
        ) = (
            new_features_113
        ) = (
            new_features_114
        ) = (
            new_features_115
        ) = (
            new_features_116
        ) = (
            new_features_117
        ) = (
            new_features_118
        ) = (
            new_features_119
        ) = (
            new_features_120
        ) = (
            new_features_121
        ) = (
            new_features_122
        ) = (
            new_features_123
        ) = (
            new_features_124
        ) = (
            new_features_125
        ) = (
            new_features_126
        ) = new_features_127 = new_features_128 = new_features_129 = None
        x_532 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_features_modules_norm5_buffers_running_mean_,
            l_self_modules_features_modules_norm5_buffers_running_var_,
            l_self_modules_features_modules_norm5_parameters_weight_,
            l_self_modules_features_modules_norm5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_14 = (
            l_self_modules_features_modules_norm5_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_norm5_buffers_running_var_
        ) = (
            l_self_modules_features_modules_norm5_parameters_weight_
        ) = l_self_modules_features_modules_norm5_parameters_bias_ = None
        x_533 = torch.nn.functional.relu(x_532, inplace=True)
        x_532 = None
        x_534 = torch.nn.functional.adaptive_avg_pool2d(x_533, 1)
        x_533 = None
        x_535 = x_534.flatten(1, -1)
        x_534 = None
        x_536 = torch.nn.functional.dropout(x_535, 0.0, False, False)
        x_535 = None
        x_537 = torch._C._nn.linear(
            x_536,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_536 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_537,)
