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
            (3, 3),
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
        input_2 = torch.nn.functional.max_pool2d(
            x_1, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_1 = None
        concated_features = torch.cat([input_2], 1)
        x_2 = torch.nn.functional.batch_norm(
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
        x_3 = torch.nn.functional.relu(x_2, inplace=True)
        x_2 = None
        bottleneck_output = torch.conv2d(
            x_3,
            l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_3 = l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_conv1_parameters_weight_ = (None)
        x_4 = torch.nn.functional.batch_norm(
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
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        new_features = torch.conv2d(
            x_5,
            l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_features_modules_denseblock1_modules_denselayer1_modules_conv2_parameters_weight_ = (None)
        concated_features_1 = torch.cat([input_2, new_features], 1)
        x_6 = torch.nn.functional.batch_norm(
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
        x_7 = torch.nn.functional.relu(x_6, inplace=True)
        x_6 = None
        bottleneck_output_1 = torch.conv2d(
            x_7,
            l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_7 = l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_conv1_parameters_weight_ = (None)
        x_8 = torch.nn.functional.batch_norm(
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
        x_9 = torch.nn.functional.relu(x_8, inplace=True)
        x_8 = None
        new_features_1 = torch.conv2d(
            x_9,
            l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_features_modules_denseblock1_modules_denselayer2_modules_conv2_parameters_weight_ = (None)
        concated_features_2 = torch.cat([input_2, new_features, new_features_1], 1)
        x_10 = torch.nn.functional.batch_norm(
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
        x_11 = torch.nn.functional.relu(x_10, inplace=True)
        x_10 = None
        bottleneck_output_2 = torch.conv2d(
            x_11,
            l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_11 = l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_conv1_parameters_weight_ = (None)
        x_12 = torch.nn.functional.batch_norm(
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
        x_13 = torch.nn.functional.relu(x_12, inplace=True)
        x_12 = None
        new_features_2 = torch.conv2d(
            x_13,
            l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_features_modules_denseblock1_modules_denselayer3_modules_conv2_parameters_weight_ = (None)
        concated_features_3 = torch.cat(
            [input_2, new_features, new_features_1, new_features_2], 1
        )
        x_14 = torch.nn.functional.batch_norm(
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
        x_15 = torch.nn.functional.relu(x_14, inplace=True)
        x_14 = None
        bottleneck_output_3 = torch.conv2d(
            x_15,
            l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_15 = l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_conv1_parameters_weight_ = (None)
        x_16 = torch.nn.functional.batch_norm(
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
        x_17 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        new_features_3 = torch.conv2d(
            x_17,
            l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_17 = l_self_modules_features_modules_denseblock1_modules_denselayer4_modules_conv2_parameters_weight_ = (None)
        concated_features_4 = torch.cat(
            [input_2, new_features, new_features_1, new_features_2, new_features_3], 1
        )
        x_18 = torch.nn.functional.batch_norm(
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
        x_19 = torch.nn.functional.relu(x_18, inplace=True)
        x_18 = None
        bottleneck_output_4 = torch.conv2d(
            x_19,
            l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_19 = l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_conv1_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
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
        x_21 = torch.nn.functional.relu(x_20, inplace=True)
        x_20 = None
        new_features_4 = torch.conv2d(
            x_21,
            l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_21 = l_self_modules_features_modules_denseblock1_modules_denselayer5_modules_conv2_parameters_weight_ = (None)
        concated_features_5 = torch.cat(
            [
                input_2,
                new_features,
                new_features_1,
                new_features_2,
                new_features_3,
                new_features_4,
            ],
            1,
        )
        x_22 = torch.nn.functional.batch_norm(
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
        x_23 = torch.nn.functional.relu(x_22, inplace=True)
        x_22 = None
        bottleneck_output_5 = torch.conv2d(
            x_23,
            l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_23 = l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_conv1_parameters_weight_ = (None)
        x_24 = torch.nn.functional.batch_norm(
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
        x_25 = torch.nn.functional.relu(x_24, inplace=True)
        x_24 = None
        new_features_5 = torch.conv2d(
            x_25,
            l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_features_modules_denseblock1_modules_denselayer6_modules_conv2_parameters_weight_ = (None)
        input_3 = torch.cat(
            [
                input_2,
                new_features,
                new_features_1,
                new_features_2,
                new_features_3,
                new_features_4,
                new_features_5,
            ],
            1,
        )
        input_2 = (
            new_features
        ) = (
            new_features_1
        ) = new_features_2 = new_features_3 = new_features_4 = new_features_5 = None
        x_26 = torch.nn.functional.batch_norm(
            input_3,
            l_self_modules_features_modules_transition1_modules_norm_buffers_running_mean_,
            l_self_modules_features_modules_transition1_modules_norm_buffers_running_var_,
            l_self_modules_features_modules_transition1_modules_norm_parameters_weight_,
            l_self_modules_features_modules_transition1_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_3 = l_self_modules_features_modules_transition1_modules_norm_buffers_running_mean_ = l_self_modules_features_modules_transition1_modules_norm_buffers_running_var_ = (
            l_self_modules_features_modules_transition1_modules_norm_parameters_weight_
        ) = (
            l_self_modules_features_modules_transition1_modules_norm_parameters_bias_
        ) = None
        x_27 = torch.nn.functional.relu(x_26, inplace=True)
        x_26 = None
        input_4 = torch.conv2d(
            x_27,
            l_self_modules_features_modules_transition1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_27 = (
            l_self_modules_features_modules_transition1_modules_conv_parameters_weight_
        ) = None
        input_5 = torch._C._nn.avg_pool2d(input_4, 2, 2, 0, False, True, None)
        input_4 = None
        concated_features_6 = torch.cat([input_5], 1)
        x_28 = torch.nn.functional.batch_norm(
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
        x_29 = torch.nn.functional.relu(x_28, inplace=True)
        x_28 = None
        bottleneck_output_6 = torch.conv2d(
            x_29,
            l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_29 = l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_conv1_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
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
        x_31 = torch.nn.functional.relu(x_30, inplace=True)
        x_30 = None
        new_features_6 = torch.conv2d(
            x_31,
            l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_31 = l_self_modules_features_modules_denseblock2_modules_denselayer1_modules_conv2_parameters_weight_ = (None)
        concated_features_7 = torch.cat([input_5, new_features_6], 1)
        x_32 = torch.nn.functional.batch_norm(
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
        x_33 = torch.nn.functional.relu(x_32, inplace=True)
        x_32 = None
        bottleneck_output_7 = torch.conv2d(
            x_33,
            l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_33 = l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_conv1_parameters_weight_ = (None)
        x_34 = torch.nn.functional.batch_norm(
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
        x_35 = torch.nn.functional.relu(x_34, inplace=True)
        x_34 = None
        new_features_7 = torch.conv2d(
            x_35,
            l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_features_modules_denseblock2_modules_denselayer2_modules_conv2_parameters_weight_ = (None)
        concated_features_8 = torch.cat([input_5, new_features_6, new_features_7], 1)
        x_36 = torch.nn.functional.batch_norm(
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
        x_37 = torch.nn.functional.relu(x_36, inplace=True)
        x_36 = None
        bottleneck_output_8 = torch.conv2d(
            x_37,
            l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_conv1_parameters_weight_ = (None)
        x_38 = torch.nn.functional.batch_norm(
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
        x_39 = torch.nn.functional.relu(x_38, inplace=True)
        x_38 = None
        new_features_8 = torch.conv2d(
            x_39,
            l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_features_modules_denseblock2_modules_denselayer3_modules_conv2_parameters_weight_ = (None)
        concated_features_9 = torch.cat(
            [input_5, new_features_6, new_features_7, new_features_8], 1
        )
        x_40 = torch.nn.functional.batch_norm(
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
        x_41 = torch.nn.functional.relu(x_40, inplace=True)
        x_40 = None
        bottleneck_output_9 = torch.conv2d(
            x_41,
            l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_41 = l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_conv1_parameters_weight_ = (None)
        x_42 = torch.nn.functional.batch_norm(
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
        x_43 = torch.nn.functional.relu(x_42, inplace=True)
        x_42 = None
        new_features_9 = torch.conv2d(
            x_43,
            l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_43 = l_self_modules_features_modules_denseblock2_modules_denselayer4_modules_conv2_parameters_weight_ = (None)
        concated_features_10 = torch.cat(
            [input_5, new_features_6, new_features_7, new_features_8, new_features_9], 1
        )
        x_44 = torch.nn.functional.batch_norm(
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
        x_45 = torch.nn.functional.relu(x_44, inplace=True)
        x_44 = None
        bottleneck_output_10 = torch.conv2d(
            x_45,
            l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_45 = l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_conv1_parameters_weight_ = (None)
        x_46 = torch.nn.functional.batch_norm(
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
        x_47 = torch.nn.functional.relu(x_46, inplace=True)
        x_46 = None
        new_features_10 = torch.conv2d(
            x_47,
            l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_47 = l_self_modules_features_modules_denseblock2_modules_denselayer5_modules_conv2_parameters_weight_ = (None)
        concated_features_11 = torch.cat(
            [
                input_5,
                new_features_6,
                new_features_7,
                new_features_8,
                new_features_9,
                new_features_10,
            ],
            1,
        )
        x_48 = torch.nn.functional.batch_norm(
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
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        bottleneck_output_11 = torch.conv2d(
            x_49,
            l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_conv1_parameters_weight_ = (None)
        x_50 = torch.nn.functional.batch_norm(
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
        x_51 = torch.nn.functional.relu(x_50, inplace=True)
        x_50 = None
        new_features_11 = torch.conv2d(
            x_51,
            l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_51 = l_self_modules_features_modules_denseblock2_modules_denselayer6_modules_conv2_parameters_weight_ = (None)
        concated_features_12 = torch.cat(
            [
                input_5,
                new_features_6,
                new_features_7,
                new_features_8,
                new_features_9,
                new_features_10,
                new_features_11,
            ],
            1,
        )
        x_52 = torch.nn.functional.batch_norm(
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
        x_53 = torch.nn.functional.relu(x_52, inplace=True)
        x_52 = None
        bottleneck_output_12 = torch.conv2d(
            x_53,
            l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_conv1_parameters_weight_ = (None)
        x_54 = torch.nn.functional.batch_norm(
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
        x_55 = torch.nn.functional.relu(x_54, inplace=True)
        x_54 = None
        new_features_12 = torch.conv2d(
            x_55,
            l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_55 = l_self_modules_features_modules_denseblock2_modules_denselayer7_modules_conv2_parameters_weight_ = (None)
        concated_features_13 = torch.cat(
            [
                input_5,
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
        x_56 = torch.nn.functional.batch_norm(
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
        x_57 = torch.nn.functional.relu(x_56, inplace=True)
        x_56 = None
        bottleneck_output_13 = torch.conv2d(
            x_57,
            l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_conv1_parameters_weight_ = (None)
        x_58 = torch.nn.functional.batch_norm(
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
        x_59 = torch.nn.functional.relu(x_58, inplace=True)
        x_58 = None
        new_features_13 = torch.conv2d(
            x_59,
            l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_59 = l_self_modules_features_modules_denseblock2_modules_denselayer8_modules_conv2_parameters_weight_ = (None)
        concated_features_14 = torch.cat(
            [
                input_5,
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
        x_60 = torch.nn.functional.batch_norm(
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
        x_61 = torch.nn.functional.relu(x_60, inplace=True)
        x_60 = None
        bottleneck_output_14 = torch.conv2d(
            x_61,
            l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_61 = l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_conv1_parameters_weight_ = (None)
        x_62 = torch.nn.functional.batch_norm(
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
        x_63 = torch.nn.functional.relu(x_62, inplace=True)
        x_62 = None
        new_features_14 = torch.conv2d(
            x_63,
            l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_features_modules_denseblock2_modules_denselayer9_modules_conv2_parameters_weight_ = (None)
        concated_features_15 = torch.cat(
            [
                input_5,
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
        x_64 = torch.nn.functional.batch_norm(
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
        x_65 = torch.nn.functional.relu(x_64, inplace=True)
        x_64 = None
        bottleneck_output_15 = torch.conv2d(
            x_65,
            l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_conv1_parameters_weight_ = (None)
        x_66 = torch.nn.functional.batch_norm(
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
        x_67 = torch.nn.functional.relu(x_66, inplace=True)
        x_66 = None
        new_features_15 = torch.conv2d(
            x_67,
            l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_features_modules_denseblock2_modules_denselayer10_modules_conv2_parameters_weight_ = (None)
        concated_features_16 = torch.cat(
            [
                input_5,
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
        x_68 = torch.nn.functional.batch_norm(
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
        x_69 = torch.nn.functional.relu(x_68, inplace=True)
        x_68 = None
        bottleneck_output_16 = torch.conv2d(
            x_69,
            l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_69 = l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_conv1_parameters_weight_ = (None)
        x_70 = torch.nn.functional.batch_norm(
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
        x_71 = torch.nn.functional.relu(x_70, inplace=True)
        x_70 = None
        new_features_16 = torch.conv2d(
            x_71,
            l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_71 = l_self_modules_features_modules_denseblock2_modules_denselayer11_modules_conv2_parameters_weight_ = (None)
        concated_features_17 = torch.cat(
            [
                input_5,
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
        x_72 = torch.nn.functional.batch_norm(
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
        x_73 = torch.nn.functional.relu(x_72, inplace=True)
        x_72 = None
        bottleneck_output_17 = torch.conv2d(
            x_73,
            l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_73 = l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_conv1_parameters_weight_ = (None)
        x_74 = torch.nn.functional.batch_norm(
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
        x_75 = torch.nn.functional.relu(x_74, inplace=True)
        x_74 = None
        new_features_17 = torch.conv2d(
            x_75,
            l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_75 = l_self_modules_features_modules_denseblock2_modules_denselayer12_modules_conv2_parameters_weight_ = (None)
        input_6 = torch.cat(
            [
                input_5,
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
        input_5 = (
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
        x_76 = torch.nn.functional.batch_norm(
            input_6,
            l_self_modules_features_modules_transition2_modules_norm_buffers_running_mean_,
            l_self_modules_features_modules_transition2_modules_norm_buffers_running_var_,
            l_self_modules_features_modules_transition2_modules_norm_parameters_weight_,
            l_self_modules_features_modules_transition2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_6 = l_self_modules_features_modules_transition2_modules_norm_buffers_running_mean_ = l_self_modules_features_modules_transition2_modules_norm_buffers_running_var_ = (
            l_self_modules_features_modules_transition2_modules_norm_parameters_weight_
        ) = (
            l_self_modules_features_modules_transition2_modules_norm_parameters_bias_
        ) = None
        x_77 = torch.nn.functional.relu(x_76, inplace=True)
        x_76 = None
        input_7 = torch.conv2d(
            x_77,
            l_self_modules_features_modules_transition2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_77 = (
            l_self_modules_features_modules_transition2_modules_conv_parameters_weight_
        ) = None
        input_8 = torch._C._nn.avg_pool2d(input_7, 2, 2, 0, False, True, None)
        input_7 = None
        concated_features_18 = torch.cat([input_8], 1)
        x_78 = torch.nn.functional.batch_norm(
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
        x_79 = torch.nn.functional.relu(x_78, inplace=True)
        x_78 = None
        bottleneck_output_18 = torch.conv2d(
            x_79,
            l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_conv1_parameters_weight_ = (None)
        x_80 = torch.nn.functional.batch_norm(
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
        x_81 = torch.nn.functional.relu(x_80, inplace=True)
        x_80 = None
        new_features_18 = torch.conv2d(
            x_81,
            l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_81 = l_self_modules_features_modules_denseblock3_modules_denselayer1_modules_conv2_parameters_weight_ = (None)
        concated_features_19 = torch.cat([input_8, new_features_18], 1)
        x_82 = torch.nn.functional.batch_norm(
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
        x_83 = torch.nn.functional.relu(x_82, inplace=True)
        x_82 = None
        bottleneck_output_19 = torch.conv2d(
            x_83,
            l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_83 = l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_conv1_parameters_weight_ = (None)
        x_84 = torch.nn.functional.batch_norm(
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
        x_85 = torch.nn.functional.relu(x_84, inplace=True)
        x_84 = None
        new_features_19 = torch.conv2d(
            x_85,
            l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_85 = l_self_modules_features_modules_denseblock3_modules_denselayer2_modules_conv2_parameters_weight_ = (None)
        concated_features_20 = torch.cat([input_8, new_features_18, new_features_19], 1)
        x_86 = torch.nn.functional.batch_norm(
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
        x_87 = torch.nn.functional.relu(x_86, inplace=True)
        x_86 = None
        bottleneck_output_20 = torch.conv2d(
            x_87,
            l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_87 = l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_conv1_parameters_weight_ = (None)
        x_88 = torch.nn.functional.batch_norm(
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
        x_89 = torch.nn.functional.relu(x_88, inplace=True)
        x_88 = None
        new_features_20 = torch.conv2d(
            x_89,
            l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_features_modules_denseblock3_modules_denselayer3_modules_conv2_parameters_weight_ = (None)
        concated_features_21 = torch.cat(
            [input_8, new_features_18, new_features_19, new_features_20], 1
        )
        x_90 = torch.nn.functional.batch_norm(
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
        x_91 = torch.nn.functional.relu(x_90, inplace=True)
        x_90 = None
        bottleneck_output_21 = torch.conv2d(
            x_91,
            l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_91 = l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_conv1_parameters_weight_ = (None)
        x_92 = torch.nn.functional.batch_norm(
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
        x_93 = torch.nn.functional.relu(x_92, inplace=True)
        x_92 = None
        new_features_21 = torch.conv2d(
            x_93,
            l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_93 = l_self_modules_features_modules_denseblock3_modules_denselayer4_modules_conv2_parameters_weight_ = (None)
        concated_features_22 = torch.cat(
            [
                input_8,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
            ],
            1,
        )
        x_94 = torch.nn.functional.batch_norm(
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
        x_95 = torch.nn.functional.relu(x_94, inplace=True)
        x_94 = None
        bottleneck_output_22 = torch.conv2d(
            x_95,
            l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_95 = l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_conv1_parameters_weight_ = (None)
        x_96 = torch.nn.functional.batch_norm(
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
        x_97 = torch.nn.functional.relu(x_96, inplace=True)
        x_96 = None
        new_features_22 = torch.conv2d(
            x_97,
            l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_features_modules_denseblock3_modules_denselayer5_modules_conv2_parameters_weight_ = (None)
        concated_features_23 = torch.cat(
            [
                input_8,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
            ],
            1,
        )
        x_98 = torch.nn.functional.batch_norm(
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
        x_99 = torch.nn.functional.relu(x_98, inplace=True)
        x_98 = None
        bottleneck_output_23 = torch.conv2d(
            x_99,
            l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_99 = l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_conv1_parameters_weight_ = (None)
        x_100 = torch.nn.functional.batch_norm(
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
        x_101 = torch.nn.functional.relu(x_100, inplace=True)
        x_100 = None
        new_features_23 = torch.conv2d(
            x_101,
            l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_101 = l_self_modules_features_modules_denseblock3_modules_denselayer6_modules_conv2_parameters_weight_ = (None)
        concated_features_24 = torch.cat(
            [
                input_8,
                new_features_18,
                new_features_19,
                new_features_20,
                new_features_21,
                new_features_22,
                new_features_23,
            ],
            1,
        )
        x_102 = torch.nn.functional.batch_norm(
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
        x_103 = torch.nn.functional.relu(x_102, inplace=True)
        x_102 = None
        bottleneck_output_24 = torch.conv2d(
            x_103,
            l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_conv1_parameters_weight_ = (None)
        x_104 = torch.nn.functional.batch_norm(
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
        x_105 = torch.nn.functional.relu(x_104, inplace=True)
        x_104 = None
        new_features_24 = torch.conv2d(
            x_105,
            l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_105 = l_self_modules_features_modules_denseblock3_modules_denselayer7_modules_conv2_parameters_weight_ = (None)
        concated_features_25 = torch.cat(
            [
                input_8,
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
        x_106 = torch.nn.functional.batch_norm(
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
        x_107 = torch.nn.functional.relu(x_106, inplace=True)
        x_106 = None
        bottleneck_output_25 = torch.conv2d(
            x_107,
            l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_107 = l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_conv1_parameters_weight_ = (None)
        x_108 = torch.nn.functional.batch_norm(
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
        x_109 = torch.nn.functional.relu(x_108, inplace=True)
        x_108 = None
        new_features_25 = torch.conv2d(
            x_109,
            l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_features_modules_denseblock3_modules_denselayer8_modules_conv2_parameters_weight_ = (None)
        concated_features_26 = torch.cat(
            [
                input_8,
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
        x_110 = torch.nn.functional.batch_norm(
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
        x_111 = torch.nn.functional.relu(x_110, inplace=True)
        x_110 = None
        bottleneck_output_26 = torch.conv2d(
            x_111,
            l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_111 = l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_conv1_parameters_weight_ = (None)
        x_112 = torch.nn.functional.batch_norm(
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
        x_113 = torch.nn.functional.relu(x_112, inplace=True)
        x_112 = None
        new_features_26 = torch.conv2d(
            x_113,
            l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_113 = l_self_modules_features_modules_denseblock3_modules_denselayer9_modules_conv2_parameters_weight_ = (None)
        concated_features_27 = torch.cat(
            [
                input_8,
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
        x_114 = torch.nn.functional.batch_norm(
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
        x_115 = torch.nn.functional.relu(x_114, inplace=True)
        x_114 = None
        bottleneck_output_27 = torch.conv2d(
            x_115,
            l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_115 = l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_conv1_parameters_weight_ = (None)
        x_116 = torch.nn.functional.batch_norm(
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
        x_117 = torch.nn.functional.relu(x_116, inplace=True)
        x_116 = None
        new_features_27 = torch.conv2d(
            x_117,
            l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_features_modules_denseblock3_modules_denselayer10_modules_conv2_parameters_weight_ = (None)
        concated_features_28 = torch.cat(
            [
                input_8,
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
        x_118 = torch.nn.functional.batch_norm(
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
        x_119 = torch.nn.functional.relu(x_118, inplace=True)
        x_118 = None
        bottleneck_output_28 = torch.conv2d(
            x_119,
            l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_119 = l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_conv1_parameters_weight_ = (None)
        x_120 = torch.nn.functional.batch_norm(
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
        x_121 = torch.nn.functional.relu(x_120, inplace=True)
        x_120 = None
        new_features_28 = torch.conv2d(
            x_121,
            l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_121 = l_self_modules_features_modules_denseblock3_modules_denselayer11_modules_conv2_parameters_weight_ = (None)
        concated_features_29 = torch.cat(
            [
                input_8,
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
        x_122 = torch.nn.functional.batch_norm(
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
        x_123 = torch.nn.functional.relu(x_122, inplace=True)
        x_122 = None
        bottleneck_output_29 = torch.conv2d(
            x_123,
            l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_123 = l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_conv1_parameters_weight_ = (None)
        x_124 = torch.nn.functional.batch_norm(
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
        x_125 = torch.nn.functional.relu(x_124, inplace=True)
        x_124 = None
        new_features_29 = torch.conv2d(
            x_125,
            l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_features_modules_denseblock3_modules_denselayer12_modules_conv2_parameters_weight_ = (None)
        concated_features_30 = torch.cat(
            [
                input_8,
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
        x_126 = torch.nn.functional.batch_norm(
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
        x_127 = torch.nn.functional.relu(x_126, inplace=True)
        x_126 = None
        bottleneck_output_30 = torch.conv2d(
            x_127,
            l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_127 = l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_conv1_parameters_weight_ = (None)
        x_128 = torch.nn.functional.batch_norm(
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
        x_129 = torch.nn.functional.relu(x_128, inplace=True)
        x_128 = None
        new_features_30 = torch.conv2d(
            x_129,
            l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_129 = l_self_modules_features_modules_denseblock3_modules_denselayer13_modules_conv2_parameters_weight_ = (None)
        concated_features_31 = torch.cat(
            [
                input_8,
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
        x_130 = torch.nn.functional.batch_norm(
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
        x_131 = torch.nn.functional.relu(x_130, inplace=True)
        x_130 = None
        bottleneck_output_31 = torch.conv2d(
            x_131,
            l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_131 = l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_conv1_parameters_weight_ = (None)
        x_132 = torch.nn.functional.batch_norm(
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
        x_133 = torch.nn.functional.relu(x_132, inplace=True)
        x_132 = None
        new_features_31 = torch.conv2d(
            x_133,
            l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_133 = l_self_modules_features_modules_denseblock3_modules_denselayer14_modules_conv2_parameters_weight_ = (None)
        concated_features_32 = torch.cat(
            [
                input_8,
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
        x_134 = torch.nn.functional.batch_norm(
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
        x_135 = torch.nn.functional.relu(x_134, inplace=True)
        x_134 = None
        bottleneck_output_32 = torch.conv2d(
            x_135,
            l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_135 = l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_conv1_parameters_weight_ = (None)
        x_136 = torch.nn.functional.batch_norm(
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
        x_137 = torch.nn.functional.relu(x_136, inplace=True)
        x_136 = None
        new_features_32 = torch.conv2d(
            x_137,
            l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_137 = l_self_modules_features_modules_denseblock3_modules_denselayer15_modules_conv2_parameters_weight_ = (None)
        concated_features_33 = torch.cat(
            [
                input_8,
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
        x_138 = torch.nn.functional.batch_norm(
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
        x_139 = torch.nn.functional.relu(x_138, inplace=True)
        x_138 = None
        bottleneck_output_33 = torch.conv2d(
            x_139,
            l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_conv1_parameters_weight_ = (None)
        x_140 = torch.nn.functional.batch_norm(
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
        x_141 = torch.nn.functional.relu(x_140, inplace=True)
        x_140 = None
        new_features_33 = torch.conv2d(
            x_141,
            l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_141 = l_self_modules_features_modules_denseblock3_modules_denselayer16_modules_conv2_parameters_weight_ = (None)
        concated_features_34 = torch.cat(
            [
                input_8,
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
        x_142 = torch.nn.functional.batch_norm(
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
        x_143 = torch.nn.functional.relu(x_142, inplace=True)
        x_142 = None
        bottleneck_output_34 = torch.conv2d(
            x_143,
            l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_143 = l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_conv1_parameters_weight_ = (None)
        x_144 = torch.nn.functional.batch_norm(
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
        x_145 = torch.nn.functional.relu(x_144, inplace=True)
        x_144 = None
        new_features_34 = torch.conv2d(
            x_145,
            l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_145 = l_self_modules_features_modules_denseblock3_modules_denselayer17_modules_conv2_parameters_weight_ = (None)
        concated_features_35 = torch.cat(
            [
                input_8,
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
        x_146 = torch.nn.functional.batch_norm(
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
        x_147 = torch.nn.functional.relu(x_146, inplace=True)
        x_146 = None
        bottleneck_output_35 = torch.conv2d(
            x_147,
            l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_147 = l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_conv1_parameters_weight_ = (None)
        x_148 = torch.nn.functional.batch_norm(
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
        x_149 = torch.nn.functional.relu(x_148, inplace=True)
        x_148 = None
        new_features_35 = torch.conv2d(
            x_149,
            l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_149 = l_self_modules_features_modules_denseblock3_modules_denselayer18_modules_conv2_parameters_weight_ = (None)
        concated_features_36 = torch.cat(
            [
                input_8,
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
        x_150 = torch.nn.functional.batch_norm(
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
        x_151 = torch.nn.functional.relu(x_150, inplace=True)
        x_150 = None
        bottleneck_output_36 = torch.conv2d(
            x_151,
            l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_151 = l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_conv1_parameters_weight_ = (None)
        x_152 = torch.nn.functional.batch_norm(
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
        x_153 = torch.nn.functional.relu(x_152, inplace=True)
        x_152 = None
        new_features_36 = torch.conv2d(
            x_153,
            l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_153 = l_self_modules_features_modules_denseblock3_modules_denselayer19_modules_conv2_parameters_weight_ = (None)
        concated_features_37 = torch.cat(
            [
                input_8,
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
        x_154 = torch.nn.functional.batch_norm(
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
        x_155 = torch.nn.functional.relu(x_154, inplace=True)
        x_154 = None
        bottleneck_output_37 = torch.conv2d(
            x_155,
            l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_155 = l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_conv1_parameters_weight_ = (None)
        x_156 = torch.nn.functional.batch_norm(
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
        x_157 = torch.nn.functional.relu(x_156, inplace=True)
        x_156 = None
        new_features_37 = torch.conv2d(
            x_157,
            l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_157 = l_self_modules_features_modules_denseblock3_modules_denselayer20_modules_conv2_parameters_weight_ = (None)
        concated_features_38 = torch.cat(
            [
                input_8,
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
        x_158 = torch.nn.functional.batch_norm(
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
        x_159 = torch.nn.functional.relu(x_158, inplace=True)
        x_158 = None
        bottleneck_output_38 = torch.conv2d(
            x_159,
            l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_159 = l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_conv1_parameters_weight_ = (None)
        x_160 = torch.nn.functional.batch_norm(
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
        x_161 = torch.nn.functional.relu(x_160, inplace=True)
        x_160 = None
        new_features_38 = torch.conv2d(
            x_161,
            l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_161 = l_self_modules_features_modules_denseblock3_modules_denselayer21_modules_conv2_parameters_weight_ = (None)
        concated_features_39 = torch.cat(
            [
                input_8,
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
        x_162 = torch.nn.functional.batch_norm(
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
        x_163 = torch.nn.functional.relu(x_162, inplace=True)
        x_162 = None
        bottleneck_output_39 = torch.conv2d(
            x_163,
            l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_163 = l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_conv1_parameters_weight_ = (None)
        x_164 = torch.nn.functional.batch_norm(
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
        x_165 = torch.nn.functional.relu(x_164, inplace=True)
        x_164 = None
        new_features_39 = torch.conv2d(
            x_165,
            l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_165 = l_self_modules_features_modules_denseblock3_modules_denselayer22_modules_conv2_parameters_weight_ = (None)
        concated_features_40 = torch.cat(
            [
                input_8,
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
        x_166 = torch.nn.functional.batch_norm(
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
        x_167 = torch.nn.functional.relu(x_166, inplace=True)
        x_166 = None
        bottleneck_output_40 = torch.conv2d(
            x_167,
            l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_167 = l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_conv1_parameters_weight_ = (None)
        x_168 = torch.nn.functional.batch_norm(
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
        x_169 = torch.nn.functional.relu(x_168, inplace=True)
        x_168 = None
        new_features_40 = torch.conv2d(
            x_169,
            l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_169 = l_self_modules_features_modules_denseblock3_modules_denselayer23_modules_conv2_parameters_weight_ = (None)
        concated_features_41 = torch.cat(
            [
                input_8,
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
        x_170 = torch.nn.functional.batch_norm(
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
        x_171 = torch.nn.functional.relu(x_170, inplace=True)
        x_170 = None
        bottleneck_output_41 = torch.conv2d(
            x_171,
            l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_171 = l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_conv1_parameters_weight_ = (None)
        x_172 = torch.nn.functional.batch_norm(
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
        x_173 = torch.nn.functional.relu(x_172, inplace=True)
        x_172 = None
        new_features_41 = torch.conv2d(
            x_173,
            l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_173 = l_self_modules_features_modules_denseblock3_modules_denselayer24_modules_conv2_parameters_weight_ = (None)
        input_9 = torch.cat(
            [
                input_8,
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
        input_8 = (
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
        x_174 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_features_modules_transition3_modules_norm_buffers_running_mean_,
            l_self_modules_features_modules_transition3_modules_norm_buffers_running_var_,
            l_self_modules_features_modules_transition3_modules_norm_parameters_weight_,
            l_self_modules_features_modules_transition3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_9 = l_self_modules_features_modules_transition3_modules_norm_buffers_running_mean_ = l_self_modules_features_modules_transition3_modules_norm_buffers_running_var_ = (
            l_self_modules_features_modules_transition3_modules_norm_parameters_weight_
        ) = (
            l_self_modules_features_modules_transition3_modules_norm_parameters_bias_
        ) = None
        x_175 = torch.nn.functional.relu(x_174, inplace=True)
        x_174 = None
        input_10 = torch.conv2d(
            x_175,
            l_self_modules_features_modules_transition3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_175 = (
            l_self_modules_features_modules_transition3_modules_conv_parameters_weight_
        ) = None
        input_11 = torch._C._nn.avg_pool2d(input_10, 2, 2, 0, False, True, None)
        input_10 = None
        concated_features_42 = torch.cat([input_11], 1)
        x_176 = torch.nn.functional.batch_norm(
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
        x_177 = torch.nn.functional.relu(x_176, inplace=True)
        x_176 = None
        bottleneck_output_42 = torch.conv2d(
            x_177,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_177 = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_conv1_parameters_weight_ = (None)
        x_178 = torch.nn.functional.batch_norm(
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
        x_179 = torch.nn.functional.relu(x_178, inplace=True)
        x_178 = None
        new_features_42 = torch.conv2d(
            x_179,
            l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_179 = l_self_modules_features_modules_denseblock4_modules_denselayer1_modules_conv2_parameters_weight_ = (None)
        concated_features_43 = torch.cat([input_11, new_features_42], 1)
        x_180 = torch.nn.functional.batch_norm(
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
        x_181 = torch.nn.functional.relu(x_180, inplace=True)
        x_180 = None
        bottleneck_output_43 = torch.conv2d(
            x_181,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_181 = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_conv1_parameters_weight_ = (None)
        x_182 = torch.nn.functional.batch_norm(
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
        x_183 = torch.nn.functional.relu(x_182, inplace=True)
        x_182 = None
        new_features_43 = torch.conv2d(
            x_183,
            l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_183 = l_self_modules_features_modules_denseblock4_modules_denselayer2_modules_conv2_parameters_weight_ = (None)
        concated_features_44 = torch.cat(
            [input_11, new_features_42, new_features_43], 1
        )
        x_184 = torch.nn.functional.batch_norm(
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
        x_185 = torch.nn.functional.relu(x_184, inplace=True)
        x_184 = None
        bottleneck_output_44 = torch.conv2d(
            x_185,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_185 = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_conv1_parameters_weight_ = (None)
        x_186 = torch.nn.functional.batch_norm(
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
        x_187 = torch.nn.functional.relu(x_186, inplace=True)
        x_186 = None
        new_features_44 = torch.conv2d(
            x_187,
            l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_187 = l_self_modules_features_modules_denseblock4_modules_denselayer3_modules_conv2_parameters_weight_ = (None)
        concated_features_45 = torch.cat(
            [input_11, new_features_42, new_features_43, new_features_44], 1
        )
        x_188 = torch.nn.functional.batch_norm(
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
        x_189 = torch.nn.functional.relu(x_188, inplace=True)
        x_188 = None
        bottleneck_output_45 = torch.conv2d(
            x_189,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_189 = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_conv1_parameters_weight_ = (None)
        x_190 = torch.nn.functional.batch_norm(
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
        x_191 = torch.nn.functional.relu(x_190, inplace=True)
        x_190 = None
        new_features_45 = torch.conv2d(
            x_191,
            l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_191 = l_self_modules_features_modules_denseblock4_modules_denselayer4_modules_conv2_parameters_weight_ = (None)
        concated_features_46 = torch.cat(
            [
                input_11,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
            ],
            1,
        )
        x_192 = torch.nn.functional.batch_norm(
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
        x_193 = torch.nn.functional.relu(x_192, inplace=True)
        x_192 = None
        bottleneck_output_46 = torch.conv2d(
            x_193,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_193 = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_conv1_parameters_weight_ = (None)
        x_194 = torch.nn.functional.batch_norm(
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
        x_195 = torch.nn.functional.relu(x_194, inplace=True)
        x_194 = None
        new_features_46 = torch.conv2d(
            x_195,
            l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_195 = l_self_modules_features_modules_denseblock4_modules_denselayer5_modules_conv2_parameters_weight_ = (None)
        concated_features_47 = torch.cat(
            [
                input_11,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
            ],
            1,
        )
        x_196 = torch.nn.functional.batch_norm(
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
        x_197 = torch.nn.functional.relu(x_196, inplace=True)
        x_196 = None
        bottleneck_output_47 = torch.conv2d(
            x_197,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_197 = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_conv1_parameters_weight_ = (None)
        x_198 = torch.nn.functional.batch_norm(
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
        x_199 = torch.nn.functional.relu(x_198, inplace=True)
        x_198 = None
        new_features_47 = torch.conv2d(
            x_199,
            l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_199 = l_self_modules_features_modules_denseblock4_modules_denselayer6_modules_conv2_parameters_weight_ = (None)
        concated_features_48 = torch.cat(
            [
                input_11,
                new_features_42,
                new_features_43,
                new_features_44,
                new_features_45,
                new_features_46,
                new_features_47,
            ],
            1,
        )
        x_200 = torch.nn.functional.batch_norm(
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
        x_201 = torch.nn.functional.relu(x_200, inplace=True)
        x_200 = None
        bottleneck_output_48 = torch.conv2d(
            x_201,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_201 = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_conv1_parameters_weight_ = (None)
        x_202 = torch.nn.functional.batch_norm(
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
        x_203 = torch.nn.functional.relu(x_202, inplace=True)
        x_202 = None
        new_features_48 = torch.conv2d(
            x_203,
            l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_203 = l_self_modules_features_modules_denseblock4_modules_denselayer7_modules_conv2_parameters_weight_ = (None)
        concated_features_49 = torch.cat(
            [
                input_11,
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
        x_204 = torch.nn.functional.batch_norm(
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
        x_205 = torch.nn.functional.relu(x_204, inplace=True)
        x_204 = None
        bottleneck_output_49 = torch.conv2d(
            x_205,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_205 = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_conv1_parameters_weight_ = (None)
        x_206 = torch.nn.functional.batch_norm(
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
        x_207 = torch.nn.functional.relu(x_206, inplace=True)
        x_206 = None
        new_features_49 = torch.conv2d(
            x_207,
            l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_207 = l_self_modules_features_modules_denseblock4_modules_denselayer8_modules_conv2_parameters_weight_ = (None)
        concated_features_50 = torch.cat(
            [
                input_11,
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
        x_208 = torch.nn.functional.batch_norm(
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
        x_209 = torch.nn.functional.relu(x_208, inplace=True)
        x_208 = None
        bottleneck_output_50 = torch.conv2d(
            x_209,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_209 = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_conv1_parameters_weight_ = (None)
        x_210 = torch.nn.functional.batch_norm(
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
        x_211 = torch.nn.functional.relu(x_210, inplace=True)
        x_210 = None
        new_features_50 = torch.conv2d(
            x_211,
            l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_211 = l_self_modules_features_modules_denseblock4_modules_denselayer9_modules_conv2_parameters_weight_ = (None)
        concated_features_51 = torch.cat(
            [
                input_11,
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
        x_212 = torch.nn.functional.batch_norm(
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
        x_213 = torch.nn.functional.relu(x_212, inplace=True)
        x_212 = None
        bottleneck_output_51 = torch.conv2d(
            x_213,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_213 = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_conv1_parameters_weight_ = (None)
        x_214 = torch.nn.functional.batch_norm(
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
        x_215 = torch.nn.functional.relu(x_214, inplace=True)
        x_214 = None
        new_features_51 = torch.conv2d(
            x_215,
            l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_215 = l_self_modules_features_modules_denseblock4_modules_denselayer10_modules_conv2_parameters_weight_ = (None)
        concated_features_52 = torch.cat(
            [
                input_11,
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
        x_216 = torch.nn.functional.batch_norm(
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
        x_217 = torch.nn.functional.relu(x_216, inplace=True)
        x_216 = None
        bottleneck_output_52 = torch.conv2d(
            x_217,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_217 = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_conv1_parameters_weight_ = (None)
        x_218 = torch.nn.functional.batch_norm(
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
        x_219 = torch.nn.functional.relu(x_218, inplace=True)
        x_218 = None
        new_features_52 = torch.conv2d(
            x_219,
            l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_219 = l_self_modules_features_modules_denseblock4_modules_denselayer11_modules_conv2_parameters_weight_ = (None)
        concated_features_53 = torch.cat(
            [
                input_11,
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
        x_220 = torch.nn.functional.batch_norm(
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
        x_221 = torch.nn.functional.relu(x_220, inplace=True)
        x_220 = None
        bottleneck_output_53 = torch.conv2d(
            x_221,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_221 = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_conv1_parameters_weight_ = (None)
        x_222 = torch.nn.functional.batch_norm(
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
        x_223 = torch.nn.functional.relu(x_222, inplace=True)
        x_222 = None
        new_features_53 = torch.conv2d(
            x_223,
            l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_223 = l_self_modules_features_modules_denseblock4_modules_denselayer12_modules_conv2_parameters_weight_ = (None)
        concated_features_54 = torch.cat(
            [
                input_11,
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
        x_224 = torch.nn.functional.batch_norm(
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
        x_225 = torch.nn.functional.relu(x_224, inplace=True)
        x_224 = None
        bottleneck_output_54 = torch.conv2d(
            x_225,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_225 = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_conv1_parameters_weight_ = (None)
        x_226 = torch.nn.functional.batch_norm(
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
        x_227 = torch.nn.functional.relu(x_226, inplace=True)
        x_226 = None
        new_features_54 = torch.conv2d(
            x_227,
            l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_227 = l_self_modules_features_modules_denseblock4_modules_denselayer13_modules_conv2_parameters_weight_ = (None)
        concated_features_55 = torch.cat(
            [
                input_11,
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
        x_228 = torch.nn.functional.batch_norm(
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
        x_229 = torch.nn.functional.relu(x_228, inplace=True)
        x_228 = None
        bottleneck_output_55 = torch.conv2d(
            x_229,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_229 = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_conv1_parameters_weight_ = (None)
        x_230 = torch.nn.functional.batch_norm(
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
        x_231 = torch.nn.functional.relu(x_230, inplace=True)
        x_230 = None
        new_features_55 = torch.conv2d(
            x_231,
            l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_231 = l_self_modules_features_modules_denseblock4_modules_denselayer14_modules_conv2_parameters_weight_ = (None)
        concated_features_56 = torch.cat(
            [
                input_11,
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
        x_232 = torch.nn.functional.batch_norm(
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
        x_233 = torch.nn.functional.relu(x_232, inplace=True)
        x_232 = None
        bottleneck_output_56 = torch.conv2d(
            x_233,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_233 = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_conv1_parameters_weight_ = (None)
        x_234 = torch.nn.functional.batch_norm(
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
        x_235 = torch.nn.functional.relu(x_234, inplace=True)
        x_234 = None
        new_features_56 = torch.conv2d(
            x_235,
            l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_235 = l_self_modules_features_modules_denseblock4_modules_denselayer15_modules_conv2_parameters_weight_ = (None)
        concated_features_57 = torch.cat(
            [
                input_11,
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
        x_236 = torch.nn.functional.batch_norm(
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
        x_237 = torch.nn.functional.relu(x_236, inplace=True)
        x_236 = None
        bottleneck_output_57 = torch.conv2d(
            x_237,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_237 = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_conv1_parameters_weight_ = (None)
        x_238 = torch.nn.functional.batch_norm(
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
        x_239 = torch.nn.functional.relu(x_238, inplace=True)
        x_238 = None
        new_features_57 = torch.conv2d(
            x_239,
            l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_239 = l_self_modules_features_modules_denseblock4_modules_denselayer16_modules_conv2_parameters_weight_ = (None)
        input_12 = torch.cat(
            [
                input_11,
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
        input_11 = (
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
        x_240 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_features_modules_norm5_buffers_running_mean_,
            l_self_modules_features_modules_norm5_buffers_running_var_,
            l_self_modules_features_modules_norm5_parameters_weight_,
            l_self_modules_features_modules_norm5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = (
            l_self_modules_features_modules_norm5_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_norm5_buffers_running_var_
        ) = (
            l_self_modules_features_modules_norm5_parameters_weight_
        ) = l_self_modules_features_modules_norm5_parameters_bias_ = None
        x_241 = torch.nn.functional.relu(x_240, inplace=True)
        x_240 = None
        x_242 = torch.nn.functional.adaptive_avg_pool2d(x_241, 1)
        x_241 = None
        x_243 = x_242.flatten(1, -1)
        x_242 = None
        x_244 = torch.nn.functional.dropout(x_243, 0.0, False, False)
        x_243 = None
        x_245 = torch._C._nn.linear(
            x_244,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_244 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_245,)
