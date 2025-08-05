import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_features_modules_conv0_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
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
        L_self_modules_features_modules_pool0_modules_1_buffers_filt_: torch.Tensor,
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
        l_self_modules_features_modules_pool0_modules_1_buffers_filt_ = (
            L_self_modules_features_modules_pool0_modules_1_buffers_filt_
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
            x_5, 3, 1, 1, 1, ceil_mode=False, return_indices=False
        )
        x_5 = None
        x_6 = torch._C._nn.pad(input_4, [1, 1, 1, 1], "reflect", None)
        input_4 = None
        input_5 = torch.conv2d(
            x_6,
            l_self_modules_features_modules_pool0_modules_1_buffers_filt_,
            stride=2,
            groups=64,
        )
        x_6 = l_self_modules_features_modules_pool0_modules_1_buffers_filt_ = None
        concated_features = torch.cat([input_5], 1)
        x_7 = torch.nn.functional.batch_norm(
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
        x_8 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        bottleneck_output = torch.conv2d(
            x_8,
            l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_8 = l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_conv1_parameters_weight_ = (None)
        x_9 = torch.nn.functional.batch_norm(
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
        x_10 = torch.nn.functional.relu(x_9, inplace=True)
        x_9 = None
        new_features = torch.conv2d(
            x_10,
            l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_10 = l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_conv2_parameters_weight_ = (None)
        concated_features_1 = torch.cat([input_5, new_features], 1)
        x_11 = torch.nn.functional.batch_norm(
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
        x_12 = torch.nn.functional.relu(x_11, inplace=True)
        x_11 = None
        bottleneck_output_1 = torch.conv2d(
            x_12,
            l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_12 = l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_conv1_parameters_weight_ = (None)
        x_13 = torch.nn.functional.batch_norm(
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
        x_14 = torch.nn.functional.relu(x_13, inplace=True)
        x_13 = None
        new_features_1 = torch.conv2d(
            x_14,
            l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_14 = l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_conv2_parameters_weight_ = (None)
        concated_features_2 = torch.cat([input_5, new_features, new_features_1], 1)
        x_15 = torch.nn.functional.batch_norm(
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
        x_16 = torch.nn.functional.relu(x_15, inplace=True)
        x_15 = None
        bottleneck_output_2 = torch.conv2d(
            x_16,
            l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_16 = l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_conv1_parameters_weight_ = (None)
        x_17 = torch.nn.functional.batch_norm(
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
        x_18 = torch.nn.functional.relu(x_17, inplace=True)
        x_17 = None
        new_features_2 = torch.conv2d(
            x_18,
            l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_conv2_parameters_weight_ = (None)
        concated_features_3 = torch.cat(
            [input_5, new_features, new_features_1, new_features_2], 1
        )
        x_19 = torch.nn.functional.batch_norm(
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
        x_20 = torch.nn.functional.relu(x_19, inplace=True)
        x_19 = None
        bottleneck_output_3 = torch.conv2d(
            x_20,
            l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_20 = l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_conv1_parameters_weight_ = (None)
        x_21 = torch.nn.functional.batch_norm(
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
        x_22 = torch.nn.functional.relu(x_21, inplace=True)
        x_21 = None
        new_features_3 = torch.conv2d(
            x_22,
            l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_22 = l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_conv2_parameters_weight_ = (None)
        concated_features_4 = torch.cat(
            [input_5, new_features, new_features_1, new_features_2, new_features_3], 1
        )
        x_23 = torch.nn.functional.batch_norm(
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
        x_24 = torch.nn.functional.relu(x_23, inplace=True)
        x_23 = None
        bottleneck_output_4 = torch.conv2d(
            x_24,
            l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_24 = l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_conv1_parameters_weight_ = (None)
        x_25 = torch.nn.functional.batch_norm(
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
        x_26 = torch.nn.functional.relu(x_25, inplace=True)
        x_25 = None
        new_features_4 = torch.conv2d(
            x_26,
            l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_26 = l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_conv2_parameters_weight_ = (None)
        concated_features_5 = torch.cat(
            [
                input_5,
                new_features,
                new_features_1,
                new_features_2,
                new_features_3,
                new_features_4,
            ],
            1,
        )
        x_27 = torch.nn.functional.batch_norm(
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
        x_28 = torch.nn.functional.relu(x_27, inplace=True)
        x_27 = None
        bottleneck_output_5 = torch.conv2d(
            x_28,
            l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_conv1_parameters_weight_ = (None)
        x_29 = torch.nn.functional.batch_norm(
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
        x_30 = torch.nn.functional.relu(x_29, inplace=True)
        x_29 = None
        new_features_5 = torch.conv2d(
            x_30,
            l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_30 = l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_conv2_parameters_weight_ = (None)
        input_6 = torch.cat(
            [
                input_5,
                new_features,
                new_features_1,
                new_features_2,
                new_features_3,
                new_features_4,
                new_features_5,
            ],
            1,
        )
        input_5 = (
            new_features
        ) = (
            new_features_1
        ) = new_features_2 = new_features_3 = new_features_4 = new_features_5 = None
        x_31 = torch.nn.functional.batch_norm(
            input_6,
            l_self_modules_features_modules_transition1_modules_norm_buffers_running_mean_,
            l_self_modules_features_modules_transition1_modules_norm_buffers_running_var_,
            l_self_modules_features_modules_transition1_modules_norm_parameters_weight_,
            l_self_modules_features_modules_transition1_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_6 = l_self_modules_features_modules_transition1_modules_norm_buffers_running_mean_ = l_self_modules_features_modules_transition1_modules_norm_buffers_running_var_ = (
            l_self_modules_features_modules_transition1_modules_norm_parameters_weight_
        ) = (
            l_self_modules_features_modules_transition1_modules_norm_parameters_bias_
        ) = None
        x_32 = torch.nn.functional.relu(x_31, inplace=True)
        x_31 = None
        input_7 = torch.conv2d(
            x_32,
            l_self_modules_features_modules_transition1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_32 = (
            l_self_modules_features_modules_transition1_modules_conv_parameters_weight_
        ) = None
        input_8 = torch._C._nn.avg_pool2d(input_7, 2, 2, 0, False, True, None)
        input_7 = None
        concated_features_6 = torch.cat([input_8], 1)
        x_33 = torch.nn.functional.batch_norm(
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
        x_34 = torch.nn.functional.relu(x_33, inplace=True)
        x_33 = None
        bottleneck_output_6 = torch.conv2d(
            x_34,
            l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_34 = l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_conv1_parameters_weight_ = (None)
        x_35 = torch.nn.functional.batch_norm(
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
        x_36 = torch.nn.functional.relu(x_35, inplace=True)
        x_35 = None
        new_features_6 = torch.conv2d(
            x_36,
            l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_36 = l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_conv2_parameters_weight_ = (None)
        concated_features_7 = torch.cat([input_8, new_features_6], 1)
        x_37 = torch.nn.functional.batch_norm(
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
        x_38 = torch.nn.functional.relu(x_37, inplace=True)
        x_37 = None
        bottleneck_output_7 = torch.conv2d(
            x_38,
            l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_38 = l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_conv1_parameters_weight_ = (None)
        x_39 = torch.nn.functional.batch_norm(
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
        x_40 = torch.nn.functional.relu(x_39, inplace=True)
        x_39 = None
        new_features_7 = torch.conv2d(
            x_40,
            l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_40 = l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_conv2_parameters_weight_ = (None)
        concated_features_8 = torch.cat([input_8, new_features_6, new_features_7], 1)
        x_41 = torch.nn.functional.batch_norm(
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
        x_42 = torch.nn.functional.relu(x_41, inplace=True)
        x_41 = None
        bottleneck_output_8 = torch.conv2d(
            x_42,
            l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_42 = l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_conv1_parameters_weight_ = (None)
        x_43 = torch.nn.functional.batch_norm(
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
        x_44 = torch.nn.functional.relu(x_43, inplace=True)
        x_43 = None
        new_features_8 = torch.conv2d(
            x_44,
            l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_44 = l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_conv2_parameters_weight_ = (None)
        concated_features_9 = torch.cat(
            [input_8, new_features_6, new_features_7, new_features_8], 1
        )
        x_45 = torch.nn.functional.batch_norm(
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
        x_46 = torch.nn.functional.relu(x_45, inplace=True)
        x_45 = None
        bottleneck_output_9 = torch.conv2d(
            x_46,
            l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_conv1_parameters_weight_ = (None)
        x_47 = torch.nn.functional.batch_norm(
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
        x_48 = torch.nn.functional.relu(x_47, inplace=True)
        x_47 = None
        new_features_9 = torch.conv2d(
            x_48,
            l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_48 = l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_conv2_parameters_weight_ = (None)
        concated_features_10 = torch.cat(
            [input_8, new_features_6, new_features_7, new_features_8, new_features_9], 1
        )
        x_49 = torch.nn.functional.batch_norm(
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
        x_50 = torch.nn.functional.relu(x_49, inplace=True)
        x_49 = None
        bottleneck_output_10 = torch.conv2d(
            x_50,
            l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_50 = l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_conv1_parameters_weight_ = (None)
        x_51 = torch.nn.functional.batch_norm(
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
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        new_features_10 = torch.conv2d(
            x_52,
            l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_conv2_parameters_weight_ = (None)
        concated_features_11 = torch.cat(
            [
                input_8,
                new_features_6,
                new_features_7,
                new_features_8,
                new_features_9,
                new_features_10,
            ],
            1,
        )
        x_53 = torch.nn.functional.batch_norm(
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
        x_54 = torch.nn.functional.relu(x_53, inplace=True)
        x_53 = None
        bottleneck_output_11 = torch.conv2d(
            x_54,
            l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_54 = l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_conv1_parameters_weight_ = (None)
        x_55 = torch.nn.functional.batch_norm(
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
        x_56 = torch.nn.functional.relu(x_55, inplace=True)
        x_55 = None
        new_features_11 = torch.conv2d(
            x_56,
            l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_56 = l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_conv2_parameters_weight_ = (None)
        concated_features_12 = torch.cat(
            [
                input_8,
                new_features_6,
                new_features_7,
                new_features_8,
                new_features_9,
                new_features_10,
                new_features_11,
            ],
            1,
        )
        x_57 = torch.nn.functional.batch_norm(
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
        x_58 = torch.nn.functional.relu(x_57, inplace=True)
        x_57 = None
        bottleneck_output_12 = torch.conv2d(
            x_58,
            l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_58 = l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_conv1_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
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
        x_60 = torch.nn.functional.relu(x_59, inplace=True)
        x_59 = None
        new_features_12 = torch.conv2d(
            x_60,
            l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_60 = l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_conv2_parameters_weight_ = (None)
        concated_features_13 = torch.cat(
            [
                input_8,
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
        x_61 = torch.nn.functional.batch_norm(
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
        x_62 = torch.nn.functional.relu(x_61, inplace=True)
        x_61 = None
        bottleneck_output_13 = torch.conv2d(
            x_62,
            l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_62 = l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_conv1_parameters_weight_ = (None)
        x_63 = torch.nn.functional.batch_norm(
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
        x_64 = torch.nn.functional.relu(x_63, inplace=True)
        x_63 = None
        new_features_13 = torch.conv2d(
            x_64,
            l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_64 = l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_conv2_parameters_weight_ = (None)
        concated_features_14 = torch.cat(
            [
                input_8,
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
        x_65 = torch.nn.functional.batch_norm(
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
        x_66 = torch.nn.functional.relu(x_65, inplace=True)
        x_65 = None
        bottleneck_output_14 = torch.conv2d(
            x_66,
            l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_66 = l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_conv1_parameters_weight_ = (None)
        x_67 = torch.nn.functional.batch_norm(
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
        x_68 = torch.nn.functional.relu(x_67, inplace=True)
        x_67 = None
        new_features_14 = torch.conv2d(
            x_68,
            l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_68 = l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_conv2_parameters_weight_ = (None)
        concated_features_15 = torch.cat(
            [
                input_8,
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
        x_69 = torch.nn.functional.batch_norm(
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
        x_70 = torch.nn.functional.relu(x_69, inplace=True)
        x_69 = None
        bottleneck_output_15 = torch.conv2d(
            x_70,
            l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_70 = l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_conv1_parameters_weight_ = (None)
        x_71 = torch.nn.functional.batch_norm(
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
        x_72 = torch.nn.functional.relu(x_71, inplace=True)
        x_71 = None
        new_features_15 = torch.conv2d(
            x_72,
            l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_72 = l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_conv2_parameters_weight_ = (None)
        concated_features_16 = torch.cat(
            [
                input_8,
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
        x_73 = torch.nn.functional.batch_norm(
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
        x_74 = torch.nn.functional.relu(x_73, inplace=True)
        x_73 = None
        bottleneck_output_16 = torch.conv2d(
            x_74,
            l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_conv1_parameters_weight_ = (None)
        x_75 = torch.nn.functional.batch_norm(
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
        x_76 = torch.nn.functional.relu(x_75, inplace=True)
        x_75 = None
        new_features_16 = torch.conv2d(
            x_76,
            l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_76 = l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_conv2_parameters_weight_ = (None)
        concated_features_17 = torch.cat(
            [
                input_8,
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
        x_77 = torch.nn.functional.batch_norm(
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
        x_78 = torch.nn.functional.relu(x_77, inplace=True)
        x_77 = None
        bottleneck_output_17 = torch.conv2d(
            x_78,
            l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_78 = l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_conv1_parameters_weight_ = (None)
        x_79 = torch.nn.functional.batch_norm(
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
        x_80 = torch.nn.functional.relu(x_79, inplace=True)
        x_79 = None
        new_features_17 = torch.conv2d(
            x_80,
            l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_80 = l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_conv2_parameters_weight_ = (None)
        input_9 = torch.cat(
            [
                input_8,
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
        input_8 = (
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
        x_81 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_features_modules_transition2_modules_norm_buffers_running_mean_,
            l_self_modules_features_modules_transition2_modules_norm_buffers_running_var_,
            l_self_modules_features_modules_transition2_modules_norm_parameters_weight_,
            l_self_modules_features_modules_transition2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_9 = l_self_modules_features_modules_transition2_modules_norm_buffers_running_mean_ = l_self_modules_features_modules_transition2_modules_norm_buffers_running_var_ = (
            l_self_modules_features_modules_transition2_modules_norm_parameters_weight_
        ) = (
            l_self_modules_features_modules_transition2_modules_norm_parameters_bias_
        ) = None
        x_82 = torch.nn.functional.relu(x_81, inplace=True)
        x_81 = None
        input_10 = torch.conv2d(
            x_82,
            l_self_modules_features_modules_transition2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_82 = (
            l_self_modules_features_modules_transition2_modules_conv_parameters_weight_
        ) = None
        input_11 = torch._C._nn.avg_pool2d(input_10, 2, 2, 0, False, True, None)
        input_10 = None
        concated_features_18 = torch.cat([input_11], 1)
        x_83 = torch.nn.functional.batch_norm(
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
        x_84 = torch.nn.functional.relu(x_83, inplace=True)
        x_83 = None
        bottleneck_output_18 = torch.conv2d(
            x_84,
            l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_84 = l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_conv1_parameters_weight_ = (None)
        x_85 = torch.nn.functional.batch_norm(
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
        x_86 = torch.nn.functional.relu(x_85, inplace=True)
        x_85 = None
        new_features_18 = torch.conv2d(
            x_86,
            l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_86 = l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_conv2_parameters_weight_ = (None)
        concated_features_19 = torch.cat([input_11, new_features_18], 1)
        x_87 = torch.nn.functional.batch_norm(
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
        x_88 = torch.nn.functional.relu(x_87, inplace=True)
        x_87 = None
        bottleneck_output_19 = torch.conv2d(
            x_88,
            l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_88 = l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_conv1_parameters_weight_ = (None)
        x_89 = torch.nn.functional.batch_norm(
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
        x_90 = torch.nn.functional.relu(x_89, inplace=True)
        x_89 = None
        new_features_19 = torch.conv2d(
            x_90,
            l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_90 = l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_conv2_parameters_weight_ = (None)
        concated_features_20 = torch.cat(
            [input_11, new_features_18, new_features_19], 1
        )
        x_91 = torch.nn.functional.batch_norm(
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
        x_92 = torch.nn.functional.relu(x_91, inplace=True)
        x_91 = None
        bottleneck_output_20 = torch.conv2d(
            x_92,
            l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_92 = l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_conv1_parameters_weight_ = (None)
        x_93 = torch.nn.functional.batch_norm(
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
        x_94 = torch.nn.functional.relu(x_93, inplace=True)
        x_93 = None
        new_features_20 = torch.conv2d(
            x_94,
            l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_94 = l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_conv2_parameters_weight_ = (None)
        concated_features_21 = torch.cat(
            [input_11, new_features_18, new_features_19, new_features_20], 1
        )
        x_95 = torch.nn.functional.batch_norm(
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
        x_96 = torch.nn.functional.relu(x_95, inplace=True)
        x_95 = None
        bottleneck_output_21 = torch.conv2d(
            x_96,
            l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_96 = l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_conv1_parameters_weight_ = (None)
        x_97 = torch.nn.functional.batch_norm(
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
        x_98 = torch.nn.functional.relu(x_97, inplace=True)
        x_97 = None
        new_features_21 = torch.conv2d(
            x_98,
            l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_98 = l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_conv2_parameters_weight_ = (None)
        concated_features_22 = torch.cat(
            [
                input_11,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
            ],
            1,
        )
        x_99 = torch.nn.functional.batch_norm(
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
        x_100 = torch.nn.functional.relu(x_99, inplace=True)
        x_99 = None
        bottleneck_output_22 = torch.conv2d(
            x_100,
            l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_100 = l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_conv1_parameters_weight_ = (None)
        x_101 = torch.nn.functional.batch_norm(
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
        x_102 = torch.nn.functional.relu(x_101, inplace=True)
        x_101 = None
        new_features_22 = torch.conv2d(
            x_102,
            l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_102 = l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_conv2_parameters_weight_ = (None)
        concated_features_23 = torch.cat(
            [
                input_11,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
            ],
            1,
        )
        x_103 = torch.nn.functional.batch_norm(
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
        x_104 = torch.nn.functional.relu(x_103, inplace=True)
        x_103 = None
        bottleneck_output_23 = torch.conv2d(
            x_104,
            l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_104 = l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_conv1_parameters_weight_ = (None)
        x_105 = torch.nn.functional.batch_norm(
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
        x_106 = torch.nn.functional.relu(x_105, inplace=True)
        x_105 = None
        new_features_23 = torch.conv2d(
            x_106,
            l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_106 = l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_conv2_parameters_weight_ = (None)
        concated_features_24 = torch.cat(
            [
                input_11,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
            ],
            1,
        )
        x_107 = torch.nn.functional.batch_norm(
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
        x_108 = torch.nn.functional.relu(x_107, inplace=True)
        x_107 = None
        bottleneck_output_24 = torch.conv2d(
            x_108,
            l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_108 = l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_conv1_parameters_weight_ = (None)
        x_109 = torch.nn.functional.batch_norm(
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
        x_110 = torch.nn.functional.relu(x_109, inplace=True)
        x_109 = None
        new_features_24 = torch.conv2d(
            x_110,
            l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_110 = l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_conv2_parameters_weight_ = (None)
        concated_features_25 = torch.cat(
            [
                input_11,
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
        x_111 = torch.nn.functional.batch_norm(
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
        x_112 = torch.nn.functional.relu(x_111, inplace=True)
        x_111 = None
        bottleneck_output_25 = torch.conv2d(
            x_112,
            l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_112 = l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_conv1_parameters_weight_ = (None)
        x_113 = torch.nn.functional.batch_norm(
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
        x_114 = torch.nn.functional.relu(x_113, inplace=True)
        x_113 = None
        new_features_25 = torch.conv2d(
            x_114,
            l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_114 = l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_conv2_parameters_weight_ = (None)
        concated_features_26 = torch.cat(
            [
                input_11,
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
        x_115 = torch.nn.functional.batch_norm(
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
        x_116 = torch.nn.functional.relu(x_115, inplace=True)
        x_115 = None
        bottleneck_output_26 = torch.conv2d(
            x_116,
            l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_116 = l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_conv1_parameters_weight_ = (None)
        x_117 = torch.nn.functional.batch_norm(
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
        x_118 = torch.nn.functional.relu(x_117, inplace=True)
        x_117 = None
        new_features_26 = torch.conv2d(
            x_118,
            l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_conv2_parameters_weight_ = (None)
        concated_features_27 = torch.cat(
            [
                input_11,
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
        x_119 = torch.nn.functional.batch_norm(
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
        x_120 = torch.nn.functional.relu(x_119, inplace=True)
        x_119 = None
        bottleneck_output_27 = torch.conv2d(
            x_120,
            l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_120 = l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_conv1_parameters_weight_ = (None)
        x_121 = torch.nn.functional.batch_norm(
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
        x_122 = torch.nn.functional.relu(x_121, inplace=True)
        x_121 = None
        new_features_27 = torch.conv2d(
            x_122,
            l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_122 = l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_conv2_parameters_weight_ = (None)
        concated_features_28 = torch.cat(
            [
                input_11,
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
        x_123 = torch.nn.functional.batch_norm(
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
        x_124 = torch.nn.functional.relu(x_123, inplace=True)
        x_123 = None
        bottleneck_output_28 = torch.conv2d(
            x_124,
            l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_124 = l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_conv1_parameters_weight_ = (None)
        x_125 = torch.nn.functional.batch_norm(
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
        x_126 = torch.nn.functional.relu(x_125, inplace=True)
        x_125 = None
        new_features_28 = torch.conv2d(
            x_126,
            l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_126 = l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_conv2_parameters_weight_ = (None)
        concated_features_29 = torch.cat(
            [
                input_11,
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
        x_127 = torch.nn.functional.batch_norm(
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
        x_128 = torch.nn.functional.relu(x_127, inplace=True)
        x_127 = None
        bottleneck_output_29 = torch.conv2d(
            x_128,
            l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_128 = l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_conv1_parameters_weight_ = (None)
        x_129 = torch.nn.functional.batch_norm(
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
        x_130 = torch.nn.functional.relu(x_129, inplace=True)
        x_129 = None
        new_features_29 = torch.conv2d(
            x_130,
            l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_130 = l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_conv2_parameters_weight_ = (None)
        concated_features_30 = torch.cat(
            [
                input_11,
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
        x_131 = torch.nn.functional.batch_norm(
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
        x_132 = torch.nn.functional.relu(x_131, inplace=True)
        x_131 = None
        bottleneck_output_30 = torch.conv2d(
            x_132,
            l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_132 = l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_conv1_parameters_weight_ = (None)
        x_133 = torch.nn.functional.batch_norm(
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
        x_134 = torch.nn.functional.relu(x_133, inplace=True)
        x_133 = None
        new_features_30 = torch.conv2d(
            x_134,
            l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_134 = l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_conv2_parameters_weight_ = (None)
        concated_features_31 = torch.cat(
            [
                input_11,
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
        x_135 = torch.nn.functional.batch_norm(
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
        x_136 = torch.nn.functional.relu(x_135, inplace=True)
        x_135 = None
        bottleneck_output_31 = torch.conv2d(
            x_136,
            l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_136 = l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_conv1_parameters_weight_ = (None)
        x_137 = torch.nn.functional.batch_norm(
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
        x_138 = torch.nn.functional.relu(x_137, inplace=True)
        x_137 = None
        new_features_31 = torch.conv2d(
            x_138,
            l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_138 = l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_conv2_parameters_weight_ = (None)
        concated_features_32 = torch.cat(
            [
                input_11,
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
        x_139 = torch.nn.functional.batch_norm(
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
        x_140 = torch.nn.functional.relu(x_139, inplace=True)
        x_139 = None
        bottleneck_output_32 = torch.conv2d(
            x_140,
            l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_140 = l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_conv1_parameters_weight_ = (None)
        x_141 = torch.nn.functional.batch_norm(
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
        x_142 = torch.nn.functional.relu(x_141, inplace=True)
        x_141 = None
        new_features_32 = torch.conv2d(
            x_142,
            l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_142 = l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_conv2_parameters_weight_ = (None)
        concated_features_33 = torch.cat(
            [
                input_11,
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
        x_143 = torch.nn.functional.batch_norm(
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
        x_144 = torch.nn.functional.relu(x_143, inplace=True)
        x_143 = None
        bottleneck_output_33 = torch.conv2d(
            x_144,
            l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_144 = l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_conv1_parameters_weight_ = (None)
        x_145 = torch.nn.functional.batch_norm(
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
        x_146 = torch.nn.functional.relu(x_145, inplace=True)
        x_145 = None
        new_features_33 = torch.conv2d(
            x_146,
            l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_146 = l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_conv2_parameters_weight_ = (None)
        concated_features_34 = torch.cat(
            [
                input_11,
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
        x_147 = torch.nn.functional.batch_norm(
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
        x_148 = torch.nn.functional.relu(x_147, inplace=True)
        x_147 = None
        bottleneck_output_34 = torch.conv2d(
            x_148,
            l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_148 = l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_conv1_parameters_weight_ = (None)
        x_149 = torch.nn.functional.batch_norm(
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
        x_150 = torch.nn.functional.relu(x_149, inplace=True)
        x_149 = None
        new_features_34 = torch.conv2d(
            x_150,
            l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_150 = l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_conv2_parameters_weight_ = (None)
        concated_features_35 = torch.cat(
            [
                input_11,
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
        x_151 = torch.nn.functional.batch_norm(
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
        x_152 = torch.nn.functional.relu(x_151, inplace=True)
        x_151 = None
        bottleneck_output_35 = torch.conv2d(
            x_152,
            l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_152 = l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_conv1_parameters_weight_ = (None)
        x_153 = torch.nn.functional.batch_norm(
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
        x_154 = torch.nn.functional.relu(x_153, inplace=True)
        x_153 = None
        new_features_35 = torch.conv2d(
            x_154,
            l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_154 = l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_conv2_parameters_weight_ = (None)
        concated_features_36 = torch.cat(
            [
                input_11,
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
        x_155 = torch.nn.functional.batch_norm(
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
        x_156 = torch.nn.functional.relu(x_155, inplace=True)
        x_155 = None
        bottleneck_output_36 = torch.conv2d(
            x_156,
            l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_156 = l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_conv1_parameters_weight_ = (None)
        x_157 = torch.nn.functional.batch_norm(
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
        x_158 = torch.nn.functional.relu(x_157, inplace=True)
        x_157 = None
        new_features_36 = torch.conv2d(
            x_158,
            l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_158 = l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_conv2_parameters_weight_ = (None)
        concated_features_37 = torch.cat(
            [
                input_11,
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
        x_159 = torch.nn.functional.batch_norm(
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
        x_160 = torch.nn.functional.relu(x_159, inplace=True)
        x_159 = None
        bottleneck_output_37 = torch.conv2d(
            x_160,
            l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_160 = l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_conv1_parameters_weight_ = (None)
        x_161 = torch.nn.functional.batch_norm(
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
        x_162 = torch.nn.functional.relu(x_161, inplace=True)
        x_161 = None
        new_features_37 = torch.conv2d(
            x_162,
            l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_162 = l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_conv2_parameters_weight_ = (None)
        concated_features_38 = torch.cat(
            [
                input_11,
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
        x_163 = torch.nn.functional.batch_norm(
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
        x_164 = torch.nn.functional.relu(x_163, inplace=True)
        x_163 = None
        bottleneck_output_38 = torch.conv2d(
            x_164,
            l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_164 = l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_conv1_parameters_weight_ = (None)
        x_165 = torch.nn.functional.batch_norm(
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
        x_166 = torch.nn.functional.relu(x_165, inplace=True)
        x_165 = None
        new_features_38 = torch.conv2d(
            x_166,
            l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_166 = l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_conv2_parameters_weight_ = (None)
        concated_features_39 = torch.cat(
            [
                input_11,
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
        x_167 = torch.nn.functional.batch_norm(
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
        x_168 = torch.nn.functional.relu(x_167, inplace=True)
        x_167 = None
        bottleneck_output_39 = torch.conv2d(
            x_168,
            l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_168 = l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_conv1_parameters_weight_ = (None)
        x_169 = torch.nn.functional.batch_norm(
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
        x_170 = torch.nn.functional.relu(x_169, inplace=True)
        x_169 = None
        new_features_39 = torch.conv2d(
            x_170,
            l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_170 = l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_conv2_parameters_weight_ = (None)
        concated_features_40 = torch.cat(
            [
                input_11,
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
        x_171 = torch.nn.functional.batch_norm(
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
        x_172 = torch.nn.functional.relu(x_171, inplace=True)
        x_171 = None
        bottleneck_output_40 = torch.conv2d(
            x_172,
            l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_172 = l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_conv1_parameters_weight_ = (None)
        x_173 = torch.nn.functional.batch_norm(
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
        x_174 = torch.nn.functional.relu(x_173, inplace=True)
        x_173 = None
        new_features_40 = torch.conv2d(
            x_174,
            l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_174 = l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_conv2_parameters_weight_ = (None)
        concated_features_41 = torch.cat(
            [
                input_11,
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
        x_175 = torch.nn.functional.batch_norm(
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
        x_176 = torch.nn.functional.relu(x_175, inplace=True)
        x_175 = None
        bottleneck_output_41 = torch.conv2d(
            x_176,
            l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_176 = l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_conv1_parameters_weight_ = (None)
        x_177 = torch.nn.functional.batch_norm(
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
        x_178 = torch.nn.functional.relu(x_177, inplace=True)
        x_177 = None
        new_features_41 = torch.conv2d(
            x_178,
            l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_178 = l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_conv2_parameters_weight_ = (None)
        input_12 = torch.cat(
            [
                input_11,
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
        input_11 = (
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
        ) = new_features_38 = new_features_39 = new_features_40 = new_features_41 = None
        x_179 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_features_modules_transition3_modules_norm_buffers_running_mean_,
            l_self_modules_features_modules_transition3_modules_norm_buffers_running_var_,
            l_self_modules_features_modules_transition3_modules_norm_parameters_weight_,
            l_self_modules_features_modules_transition3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = l_self_modules_features_modules_transition3_modules_norm_buffers_running_mean_ = l_self_modules_features_modules_transition3_modules_norm_buffers_running_var_ = (
            l_self_modules_features_modules_transition3_modules_norm_parameters_weight_
        ) = (
            l_self_modules_features_modules_transition3_modules_norm_parameters_bias_
        ) = None
        x_180 = torch.nn.functional.relu(x_179, inplace=True)
        x_179 = None
        input_13 = torch.conv2d(
            x_180,
            l_self_modules_features_modules_transition3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_180 = (
            l_self_modules_features_modules_transition3_modules_conv_parameters_weight_
        ) = None
        input_14 = torch._C._nn.avg_pool2d(input_13, 2, 2, 0, False, True, None)
        input_13 = None
        concated_features_42 = torch.cat([input_14], 1)
        x_181 = torch.nn.functional.batch_norm(
            concated_features_42,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_42 = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm1_parameters_bias_ = (None)
        x_182 = torch.nn.functional.relu(x_181, inplace=True)
        x_181 = None
        bottleneck_output_42 = torch.conv2d(
            x_182,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_182 = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_conv1_parameters_weight_ = (None)
        x_183 = torch.nn.functional.batch_norm(
            bottleneck_output_42,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_42 = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_norm2_parameters_bias_ = (None)
        x_184 = torch.nn.functional.relu(x_183, inplace=True)
        x_183 = None
        new_features_42 = torch.conv2d(
            x_184,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_184 = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_conv2_parameters_weight_ = (None)
        concated_features_43 = torch.cat([input_14, new_features_42], 1)
        x_185 = torch.nn.functional.batch_norm(
            concated_features_43,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_43 = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm1_parameters_bias_ = (None)
        x_186 = torch.nn.functional.relu(x_185, inplace=True)
        x_185 = None
        bottleneck_output_43 = torch.conv2d(
            x_186,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_186 = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_conv1_parameters_weight_ = (None)
        x_187 = torch.nn.functional.batch_norm(
            bottleneck_output_43,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_43 = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_norm2_parameters_bias_ = (None)
        x_188 = torch.nn.functional.relu(x_187, inplace=True)
        x_187 = None
        new_features_43 = torch.conv2d(
            x_188,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_188 = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_conv2_parameters_weight_ = (None)
        concated_features_44 = torch.cat(
            [input_14, new_features_42, new_features_43], 1
        )
        x_189 = torch.nn.functional.batch_norm(
            concated_features_44,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_44 = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm1_parameters_bias_ = (None)
        x_190 = torch.nn.functional.relu(x_189, inplace=True)
        x_189 = None
        bottleneck_output_44 = torch.conv2d(
            x_190,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_190 = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_conv1_parameters_weight_ = (None)
        x_191 = torch.nn.functional.batch_norm(
            bottleneck_output_44,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_44 = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_norm2_parameters_bias_ = (None)
        x_192 = torch.nn.functional.relu(x_191, inplace=True)
        x_191 = None
        new_features_44 = torch.conv2d(
            x_192,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_192 = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_conv2_parameters_weight_ = (None)
        concated_features_45 = torch.cat(
            [input_14, new_features_42, new_features_43, new_features_44], 1
        )
        x_193 = torch.nn.functional.batch_norm(
            concated_features_45,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_45 = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm1_parameters_bias_ = (None)
        x_194 = torch.nn.functional.relu(x_193, inplace=True)
        x_193 = None
        bottleneck_output_45 = torch.conv2d(
            x_194,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_194 = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_conv1_parameters_weight_ = (None)
        x_195 = torch.nn.functional.batch_norm(
            bottleneck_output_45,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_45 = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_norm2_parameters_bias_ = (None)
        x_196 = torch.nn.functional.relu(x_195, inplace=True)
        x_195 = None
        new_features_45 = torch.conv2d(
            x_196,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_196 = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_conv2_parameters_weight_ = (None)
        concated_features_46 = torch.cat(
            [
                input_14,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
            ],
            1,
        )
        x_197 = torch.nn.functional.batch_norm(
            concated_features_46,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_46 = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm1_parameters_bias_ = (None)
        x_198 = torch.nn.functional.relu(x_197, inplace=True)
        x_197 = None
        bottleneck_output_46 = torch.conv2d(
            x_198,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_198 = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_conv1_parameters_weight_ = (None)
        x_199 = torch.nn.functional.batch_norm(
            bottleneck_output_46,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_46 = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_norm2_parameters_bias_ = (None)
        x_200 = torch.nn.functional.relu(x_199, inplace=True)
        x_199 = None
        new_features_46 = torch.conv2d(
            x_200,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_200 = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_conv2_parameters_weight_ = (None)
        concated_features_47 = torch.cat(
            [
                input_14,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
            ],
            1,
        )
        x_201 = torch.nn.functional.batch_norm(
            concated_features_47,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_47 = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm1_parameters_bias_ = (None)
        x_202 = torch.nn.functional.relu(x_201, inplace=True)
        x_201 = None
        bottleneck_output_47 = torch.conv2d(
            x_202,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_202 = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_conv1_parameters_weight_ = (None)
        x_203 = torch.nn.functional.batch_norm(
            bottleneck_output_47,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_47 = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_norm2_parameters_bias_ = (None)
        x_204 = torch.nn.functional.relu(x_203, inplace=True)
        x_203 = None
        new_features_47 = torch.conv2d(
            x_204,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_204 = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_conv2_parameters_weight_ = (None)
        concated_features_48 = torch.cat(
            [
                input_14,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
            ],
            1,
        )
        x_205 = torch.nn.functional.batch_norm(
            concated_features_48,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_48 = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm1_parameters_bias_ = (None)
        x_206 = torch.nn.functional.relu(x_205, inplace=True)
        x_205 = None
        bottleneck_output_48 = torch.conv2d(
            x_206,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_206 = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_conv1_parameters_weight_ = (None)
        x_207 = torch.nn.functional.batch_norm(
            bottleneck_output_48,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_48 = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_norm2_parameters_bias_ = (None)
        x_208 = torch.nn.functional.relu(x_207, inplace=True)
        x_207 = None
        new_features_48 = torch.conv2d(
            x_208,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_208 = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_conv2_parameters_weight_ = (None)
        concated_features_49 = torch.cat(
            [
                input_14,
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
        x_209 = torch.nn.functional.batch_norm(
            concated_features_49,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_49 = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm1_parameters_bias_ = (None)
        x_210 = torch.nn.functional.relu(x_209, inplace=True)
        x_209 = None
        bottleneck_output_49 = torch.conv2d(
            x_210,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_210 = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_conv1_parameters_weight_ = (None)
        x_211 = torch.nn.functional.batch_norm(
            bottleneck_output_49,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_49 = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_norm2_parameters_bias_ = (None)
        x_212 = torch.nn.functional.relu(x_211, inplace=True)
        x_211 = None
        new_features_49 = torch.conv2d(
            x_212,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_212 = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_conv2_parameters_weight_ = (None)
        concated_features_50 = torch.cat(
            [
                input_14,
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
        x_213 = torch.nn.functional.batch_norm(
            concated_features_50,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_50 = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm1_parameters_bias_ = (None)
        x_214 = torch.nn.functional.relu(x_213, inplace=True)
        x_213 = None
        bottleneck_output_50 = torch.conv2d(
            x_214,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_214 = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_conv1_parameters_weight_ = (None)
        x_215 = torch.nn.functional.batch_norm(
            bottleneck_output_50,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_50 = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_norm2_parameters_bias_ = (None)
        x_216 = torch.nn.functional.relu(x_215, inplace=True)
        x_215 = None
        new_features_50 = torch.conv2d(
            x_216,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_216 = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_conv2_parameters_weight_ = (None)
        concated_features_51 = torch.cat(
            [
                input_14,
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
        x_217 = torch.nn.functional.batch_norm(
            concated_features_51,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_51 = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm1_parameters_bias_ = (None)
        x_218 = torch.nn.functional.relu(x_217, inplace=True)
        x_217 = None
        bottleneck_output_51 = torch.conv2d(
            x_218,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_218 = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_conv1_parameters_weight_ = (None)
        x_219 = torch.nn.functional.batch_norm(
            bottleneck_output_51,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_51 = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_norm2_parameters_bias_ = (None)
        x_220 = torch.nn.functional.relu(x_219, inplace=True)
        x_219 = None
        new_features_51 = torch.conv2d(
            x_220,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_220 = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_conv2_parameters_weight_ = (None)
        concated_features_52 = torch.cat(
            [
                input_14,
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
        x_221 = torch.nn.functional.batch_norm(
            concated_features_52,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_52 = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm1_parameters_bias_ = (None)
        x_222 = torch.nn.functional.relu(x_221, inplace=True)
        x_221 = None
        bottleneck_output_52 = torch.conv2d(
            x_222,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_222 = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_conv1_parameters_weight_ = (None)
        x_223 = torch.nn.functional.batch_norm(
            bottleneck_output_52,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_52 = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_norm2_parameters_bias_ = (None)
        x_224 = torch.nn.functional.relu(x_223, inplace=True)
        x_223 = None
        new_features_52 = torch.conv2d(
            x_224,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_224 = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_conv2_parameters_weight_ = (None)
        concated_features_53 = torch.cat(
            [
                input_14,
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
        x_225 = torch.nn.functional.batch_norm(
            concated_features_53,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_53 = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm1_parameters_bias_ = (None)
        x_226 = torch.nn.functional.relu(x_225, inplace=True)
        x_225 = None
        bottleneck_output_53 = torch.conv2d(
            x_226,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_226 = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_conv1_parameters_weight_ = (None)
        x_227 = torch.nn.functional.batch_norm(
            bottleneck_output_53,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_53 = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_norm2_parameters_bias_ = (None)
        x_228 = torch.nn.functional.relu(x_227, inplace=True)
        x_227 = None
        new_features_53 = torch.conv2d(
            x_228,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_228 = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_conv2_parameters_weight_ = (None)
        concated_features_54 = torch.cat(
            [
                input_14,
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
        x_229 = torch.nn.functional.batch_norm(
            concated_features_54,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_54 = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm1_parameters_bias_ = (None)
        x_230 = torch.nn.functional.relu(x_229, inplace=True)
        x_229 = None
        bottleneck_output_54 = torch.conv2d(
            x_230,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_230 = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_conv1_parameters_weight_ = (None)
        x_231 = torch.nn.functional.batch_norm(
            bottleneck_output_54,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_54 = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_norm2_parameters_bias_ = (None)
        x_232 = torch.nn.functional.relu(x_231, inplace=True)
        x_231 = None
        new_features_54 = torch.conv2d(
            x_232,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_232 = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_conv2_parameters_weight_ = (None)
        concated_features_55 = torch.cat(
            [
                input_14,
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
        x_233 = torch.nn.functional.batch_norm(
            concated_features_55,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_55 = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm1_parameters_bias_ = (None)
        x_234 = torch.nn.functional.relu(x_233, inplace=True)
        x_233 = None
        bottleneck_output_55 = torch.conv2d(
            x_234,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_234 = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_conv1_parameters_weight_ = (None)
        x_235 = torch.nn.functional.batch_norm(
            bottleneck_output_55,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_55 = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_norm2_parameters_bias_ = (None)
        x_236 = torch.nn.functional.relu(x_235, inplace=True)
        x_235 = None
        new_features_55 = torch.conv2d(
            x_236,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_236 = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_conv2_parameters_weight_ = (None)
        concated_features_56 = torch.cat(
            [
                input_14,
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
        x_237 = torch.nn.functional.batch_norm(
            concated_features_56,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_56 = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm1_parameters_bias_ = (None)
        x_238 = torch.nn.functional.relu(x_237, inplace=True)
        x_237 = None
        bottleneck_output_56 = torch.conv2d(
            x_238,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_238 = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_conv1_parameters_weight_ = (None)
        x_239 = torch.nn.functional.batch_norm(
            bottleneck_output_56,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_56 = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_norm2_parameters_bias_ = (None)
        x_240 = torch.nn.functional.relu(x_239, inplace=True)
        x_239 = None
        new_features_56 = torch.conv2d(
            x_240,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_240 = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_conv2_parameters_weight_ = (None)
        concated_features_57 = torch.cat(
            [
                input_14,
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
        x_241 = torch.nn.functional.batch_norm(
            concated_features_57,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        concated_features_57 = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm1_parameters_bias_ = (None)
        x_242 = torch.nn.functional.relu(x_241, inplace=True)
        x_241 = None
        bottleneck_output_57 = torch.conv2d(
            x_242,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_242 = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_conv1_parameters_weight_ = (None)
        x_243 = torch.nn.functional.batch_norm(
            bottleneck_output_57,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_buffers_running_mean_,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_buffers_running_var_,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_parameters_weight_,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        bottleneck_output_57 = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_buffers_running_mean_ = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_buffers_running_var_ = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_parameters_weight_ = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_norm2_parameters_bias_ = (None)
        x_244 = torch.nn.functional.relu(x_243, inplace=True)
        x_243 = None
        new_features_57 = torch.conv2d(
            x_244,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_244 = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_conv2_parameters_weight_ = (None)
        input_15 = torch.cat(
            [
                input_14,
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
        input_14 = (
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
        ) = new_features_54 = new_features_55 = new_features_56 = new_features_57 = None
        x_245 = torch.nn.functional.batch_norm(
            input_15,
            l_self_modules_features_modules_norm5_buffers_running_mean_,
            l_self_modules_features_modules_norm5_buffers_running_var_,
            l_self_modules_features_modules_norm5_parameters_weight_,
            l_self_modules_features_modules_norm5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_15 = (
            l_self_modules_features_modules_norm5_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_norm5_buffers_running_var_
        ) = (
            l_self_modules_features_modules_norm5_parameters_weight_
        ) = l_self_modules_features_modules_norm5_parameters_bias_ = None
        x_246 = torch.nn.functional.relu(x_245, inplace=True)
        x_245 = None
        x_247 = torch.nn.functional.adaptive_avg_pool2d(x_246, 1)
        x_246 = None
        x_248 = x_247.flatten(1, -1)
        x_247 = None
        x_249 = torch.nn.functional.dropout(x_248, 0.0, False, False)
        x_248 = None
        x_250 = torch._C._nn.linear(
            x_249,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_249 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_250,)
