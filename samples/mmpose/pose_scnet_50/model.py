import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_backbone_modules_conv1_parameters_weight_ = (
            L_self_modules_backbone_modules_conv1_parameters_weight_
        )
        l_self_modules_backbone_modules_bn1_buffers_running_mean_ = (
            L_self_modules_backbone_modules_bn1_buffers_running_mean_
        )
        l_self_modules_backbone_modules_bn1_buffers_running_var_ = (
            L_self_modules_backbone_modules_bn1_buffers_running_var_
        )
        l_self_modules_backbone_modules_bn1_parameters_weight_ = (
            L_self_modules_backbone_modules_bn1_parameters_weight_
        )
        l_self_modules_backbone_modules_bn1_parameters_bias_ = (
            L_self_modules_backbone_modules_bn1_parameters_bias_
        )
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_
        l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_ = L_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_
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
            l_self_modules_backbone_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (3, 3),
            (1, 1),
            1,
        )
        l_inputs_ = l_self_modules_backbone_modules_conv1_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_backbone_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_backbone_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_bn1_parameters_weight_
        ) = l_self_modules_backbone_modules_bn1_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.nn.functional.max_pool2d(
            x_2, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_2 = None
        out_a = torch.conv2d(
            x_3,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        out_a_1 = torch.nn.functional.batch_norm(
            out_a,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a = l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_ = (None)
        out_a_2 = torch.nn.functional.relu(out_a_1, inplace=True)
        out_a_1 = None
        input_1 = torch.conv2d(
            out_a_2,
            l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_2 = l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_0_parameters_weight_ = (None)
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_k1_modules_1_parameters_bias_ = (None)
        input_3 = torch.nn.functional.relu(input_2, inplace=True)
        input_2 = None
        out_b = torch.conv2d(
            x_3,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_1 = torch.nn.functional.batch_norm(
            out_b,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b = l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_ = (None)
        out_b_2 = torch.nn.functional.relu(out_b_1, inplace=True)
        out_b_1 = None
        input_4 = torch._C._nn.avg_pool2d(out_b_2, 4, 4, 0, False, True, None)
        input_5 = torch.conv2d(
            input_4,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_4 = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_6 = torch.nn.functional.batch_norm(
            input_5,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_5 = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate = torch.nn.functional.interpolate(input_6, (64, 48))
        input_6 = None
        add = torch.add(out_b_2, interpolate)
        interpolate = None
        out = torch.sigmoid(add)
        add = None
        input_7 = torch.conv2d(
            out_b_2,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_2 = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_1 = torch.mul(input_8, out)
        input_8 = out = None
        input_9 = torch.conv2d(
            out_1,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_1 = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_10 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_9 = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_11 = torch.nn.functional.relu(input_10, inplace=True)
        input_10 = None
        cat = torch.cat([input_3, input_11], dim=1)
        input_3 = input_11 = None
        out_2 = torch.conv2d(
            cat,
            l_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat = l_self_modules_backbone_modules_layer1_modules_0_modules_conv3_parameters_weight_ = (None)
        out_3 = torch.nn.functional.batch_norm(
            out_2,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_2 = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_bn3_parameters_bias_ = (None)
        input_12 = torch.conv2d(
            x_3,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_3 = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_13 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_3 += input_13
        out_4 = out_3
        out_3 = input_13 = None
        out_5 = torch.nn.functional.relu(out_4, inplace=True)
        out_4 = None
        out_a_3 = torch.conv2d(
            out_5,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        out_a_4 = torch.nn.functional.batch_norm(
            out_a_3,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_3 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_ = (None)
        out_a_5 = torch.nn.functional.relu(out_a_4, inplace=True)
        out_a_4 = None
        input_14 = torch.conv2d(
            out_a_5,
            l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_5 = l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_0_parameters_weight_ = (None)
        input_15 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_14 = l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_k1_modules_1_parameters_bias_ = (None)
        input_16 = torch.nn.functional.relu(input_15, inplace=True)
        input_15 = None
        out_b_3 = torch.conv2d(
            out_5,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_4 = torch.nn.functional.batch_norm(
            out_b_3,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_3 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_ = (None)
        out_b_5 = torch.nn.functional.relu(out_b_4, inplace=True)
        out_b_4 = None
        input_17 = torch._C._nn.avg_pool2d(out_b_5, 4, 4, 0, False, True, None)
        input_18 = torch.conv2d(
            input_17,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_17 = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_19 = torch.nn.functional.batch_norm(
            input_18,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_18 = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_1 = torch.nn.functional.interpolate(input_19, (64, 48))
        input_19 = None
        add_1 = torch.add(out_b_5, interpolate_1)
        interpolate_1 = None
        out_6 = torch.sigmoid(add_1)
        add_1 = None
        input_20 = torch.conv2d(
            out_b_5,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_5 = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_21 = torch.nn.functional.batch_norm(
            input_20,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_20 = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_7 = torch.mul(input_21, out_6)
        input_21 = out_6 = None
        input_22 = torch.conv2d(
            out_7,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_7 = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_23 = torch.nn.functional.batch_norm(
            input_22,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_22 = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_24 = torch.nn.functional.relu(input_23, inplace=True)
        input_23 = None
        cat_1 = torch.cat([input_16, input_24], dim=1)
        input_16 = input_24 = None
        out_8 = torch.conv2d(
            cat_1,
            l_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_1 = l_self_modules_backbone_modules_layer1_modules_1_modules_conv3_parameters_weight_ = (None)
        out_9 = torch.nn.functional.batch_norm(
            out_8,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_8 = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_1_modules_bn3_parameters_bias_ = (None)
        out_9 += out_5
        out_10 = out_9
        out_9 = out_5 = None
        out_11 = torch.nn.functional.relu(out_10, inplace=True)
        out_10 = None
        out_a_6 = torch.conv2d(
            out_11,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv1_parameters_weight_ = (
            None
        )
        out_a_7 = torch.nn.functional.batch_norm(
            out_a_6,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_6 = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn1_parameters_bias_ = (None)
        out_a_8 = torch.nn.functional.relu(out_a_7, inplace=True)
        out_a_7 = None
        input_25 = torch.conv2d(
            out_a_8,
            l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_8 = l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_0_parameters_weight_ = (None)
        input_26 = torch.nn.functional.batch_norm(
            input_25,
            l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_25 = l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_k1_modules_1_parameters_bias_ = (None)
        input_27 = torch.nn.functional.relu(input_26, inplace=True)
        input_26 = None
        out_b_6 = torch.conv2d(
            out_11,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer1_modules_2_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_7 = torch.nn.functional.batch_norm(
            out_b_6,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_6 = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn2_parameters_bias_ = (None)
        out_b_8 = torch.nn.functional.relu(out_b_7, inplace=True)
        out_b_7 = None
        input_28 = torch._C._nn.avg_pool2d(out_b_8, 4, 4, 0, False, True, None)
        input_29 = torch.conv2d(
            input_28,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_28 = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_30 = torch.nn.functional.batch_norm(
            input_29,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_29 = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_2 = torch.nn.functional.interpolate(input_30, (64, 48))
        input_30 = None
        add_2 = torch.add(out_b_8, interpolate_2)
        interpolate_2 = None
        out_12 = torch.sigmoid(add_2)
        add_2 = None
        input_31 = torch.conv2d(
            out_b_8,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_8 = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_32 = torch.nn.functional.batch_norm(
            input_31,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_31 = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_13 = torch.mul(input_32, out_12)
        input_32 = out_12 = None
        input_33 = torch.conv2d(
            out_13,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_13 = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_34 = torch.nn.functional.batch_norm(
            input_33,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_33 = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_35 = torch.nn.functional.relu(input_34, inplace=True)
        input_34 = None
        cat_2 = torch.cat([input_27, input_35], dim=1)
        input_27 = input_35 = None
        out_14 = torch.conv2d(
            cat_2,
            l_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_2 = l_self_modules_backbone_modules_layer1_modules_2_modules_conv3_parameters_weight_ = (None)
        out_15 = torch.nn.functional.batch_norm(
            out_14,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_14 = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer1_modules_2_modules_bn3_parameters_bias_ = (None)
        out_15 += out_11
        out_16 = out_15
        out_15 = out_11 = None
        out_17 = torch.nn.functional.relu(out_16, inplace=True)
        out_16 = None
        out_a_9 = torch.conv2d(
            out_17,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        out_a_10 = torch.nn.functional.batch_norm(
            out_a_9,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_9 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_ = (None)
        out_a_11 = torch.nn.functional.relu(out_a_10, inplace=True)
        out_a_10 = None
        input_36 = torch.conv2d(
            out_a_11,
            l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_11 = l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_0_parameters_weight_ = (None)
        input_37 = torch.nn.functional.batch_norm(
            input_36,
            l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_36 = l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_k1_modules_1_parameters_bias_ = (None)
        input_38 = torch.nn.functional.relu(input_37, inplace=True)
        input_37 = None
        out_b_9 = torch.conv2d(
            out_17,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_10 = torch.nn.functional.batch_norm(
            out_b_9,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_9 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_ = (None)
        out_b_11 = torch.nn.functional.relu(out_b_10, inplace=True)
        out_b_10 = None
        input_39 = torch._C._nn.avg_pool2d(out_b_11, 4, 4, 0, False, True, None)
        input_40 = torch.conv2d(
            input_39,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_39 = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_41 = torch.nn.functional.batch_norm(
            input_40,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_40 = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_3 = torch.nn.functional.interpolate(input_41, (64, 48))
        input_41 = None
        add_3 = torch.add(out_b_11, interpolate_3)
        interpolate_3 = None
        out_18 = torch.sigmoid(add_3)
        add_3 = None
        input_42 = torch.conv2d(
            out_b_11,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_11 = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_43 = torch.nn.functional.batch_norm(
            input_42,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_42 = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_19 = torch.mul(input_43, out_18)
        input_43 = out_18 = None
        input_44 = torch.conv2d(
            out_19,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        out_19 = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_45 = torch.nn.functional.batch_norm(
            input_44,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_44 = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_46 = torch.nn.functional.relu(input_45, inplace=True)
        input_45 = None
        cat_3 = torch.cat([input_38, input_46], dim=1)
        input_38 = input_46 = None
        out_20 = torch.conv2d(
            cat_3,
            l_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_3 = l_self_modules_backbone_modules_layer2_modules_0_modules_conv3_parameters_weight_ = (None)
        out_21 = torch.nn.functional.batch_norm(
            out_20,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_20 = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_bn3_parameters_bias_ = (None)
        input_47 = torch.conv2d(
            out_17,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_17 = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_48 = torch.nn.functional.batch_norm(
            input_47,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_47 = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_21 += input_48
        out_22 = out_21
        out_21 = input_48 = None
        out_23 = torch.nn.functional.relu(out_22, inplace=True)
        out_22 = None
        out_a_12 = torch.conv2d(
            out_23,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        out_a_13 = torch.nn.functional.batch_norm(
            out_a_12,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_12 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_ = (None)
        out_a_14 = torch.nn.functional.relu(out_a_13, inplace=True)
        out_a_13 = None
        input_49 = torch.conv2d(
            out_a_14,
            l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_14 = l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_0_parameters_weight_ = (None)
        input_50 = torch.nn.functional.batch_norm(
            input_49,
            l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_49 = l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_k1_modules_1_parameters_bias_ = (None)
        input_51 = torch.nn.functional.relu(input_50, inplace=True)
        input_50 = None
        out_b_12 = torch.conv2d(
            out_23,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_13 = torch.nn.functional.batch_norm(
            out_b_12,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_12 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_ = (None)
        out_b_14 = torch.nn.functional.relu(out_b_13, inplace=True)
        out_b_13 = None
        input_52 = torch._C._nn.avg_pool2d(out_b_14, 4, 4, 0, False, True, None)
        input_53 = torch.conv2d(
            input_52,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_52 = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_54 = torch.nn.functional.batch_norm(
            input_53,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_53 = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_4 = torch.nn.functional.interpolate(input_54, (32, 24))
        input_54 = None
        add_4 = torch.add(out_b_14, interpolate_4)
        interpolate_4 = None
        out_24 = torch.sigmoid(add_4)
        add_4 = None
        input_55 = torch.conv2d(
            out_b_14,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_14 = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_56 = torch.nn.functional.batch_norm(
            input_55,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_55 = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_25 = torch.mul(input_56, out_24)
        input_56 = out_24 = None
        input_57 = torch.conv2d(
            out_25,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_25 = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_58 = torch.nn.functional.batch_norm(
            input_57,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_57 = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_59 = torch.nn.functional.relu(input_58, inplace=True)
        input_58 = None
        cat_4 = torch.cat([input_51, input_59], dim=1)
        input_51 = input_59 = None
        out_26 = torch.conv2d(
            cat_4,
            l_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_4 = l_self_modules_backbone_modules_layer2_modules_1_modules_conv3_parameters_weight_ = (None)
        out_27 = torch.nn.functional.batch_norm(
            out_26,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_26 = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_1_modules_bn3_parameters_bias_ = (None)
        out_27 += out_23
        out_28 = out_27
        out_27 = out_23 = None
        out_29 = torch.nn.functional.relu(out_28, inplace=True)
        out_28 = None
        out_a_15 = torch.conv2d(
            out_29,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv1_parameters_weight_ = (
            None
        )
        out_a_16 = torch.nn.functional.batch_norm(
            out_a_15,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_15 = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn1_parameters_bias_ = (None)
        out_a_17 = torch.nn.functional.relu(out_a_16, inplace=True)
        out_a_16 = None
        input_60 = torch.conv2d(
            out_a_17,
            l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_17 = l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_0_parameters_weight_ = (None)
        input_61 = torch.nn.functional.batch_norm(
            input_60,
            l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_60 = l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_k1_modules_1_parameters_bias_ = (None)
        input_62 = torch.nn.functional.relu(input_61, inplace=True)
        input_61 = None
        out_b_15 = torch.conv2d(
            out_29,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_2_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_16 = torch.nn.functional.batch_norm(
            out_b_15,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_15 = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn2_parameters_bias_ = (None)
        out_b_17 = torch.nn.functional.relu(out_b_16, inplace=True)
        out_b_16 = None
        input_63 = torch._C._nn.avg_pool2d(out_b_17, 4, 4, 0, False, True, None)
        input_64 = torch.conv2d(
            input_63,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_63 = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_65 = torch.nn.functional.batch_norm(
            input_64,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_64 = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_5 = torch.nn.functional.interpolate(input_65, (32, 24))
        input_65 = None
        add_5 = torch.add(out_b_17, interpolate_5)
        interpolate_5 = None
        out_30 = torch.sigmoid(add_5)
        add_5 = None
        input_66 = torch.conv2d(
            out_b_17,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_17 = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_67 = torch.nn.functional.batch_norm(
            input_66,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_66 = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_31 = torch.mul(input_67, out_30)
        input_67 = out_30 = None
        input_68 = torch.conv2d(
            out_31,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_31 = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_69 = torch.nn.functional.batch_norm(
            input_68,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_68 = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_70 = torch.nn.functional.relu(input_69, inplace=True)
        input_69 = None
        cat_5 = torch.cat([input_62, input_70], dim=1)
        input_62 = input_70 = None
        out_32 = torch.conv2d(
            cat_5,
            l_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_5 = l_self_modules_backbone_modules_layer2_modules_2_modules_conv3_parameters_weight_ = (None)
        out_33 = torch.nn.functional.batch_norm(
            out_32,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_32 = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_2_modules_bn3_parameters_bias_ = (None)
        out_33 += out_29
        out_34 = out_33
        out_33 = out_29 = None
        out_35 = torch.nn.functional.relu(out_34, inplace=True)
        out_34 = None
        out_a_18 = torch.conv2d(
            out_35,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv1_parameters_weight_ = (
            None
        )
        out_a_19 = torch.nn.functional.batch_norm(
            out_a_18,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_18 = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn1_parameters_bias_ = (None)
        out_a_20 = torch.nn.functional.relu(out_a_19, inplace=True)
        out_a_19 = None
        input_71 = torch.conv2d(
            out_a_20,
            l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_20 = l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_0_parameters_weight_ = (None)
        input_72 = torch.nn.functional.batch_norm(
            input_71,
            l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_71 = l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_k1_modules_1_parameters_bias_ = (None)
        input_73 = torch.nn.functional.relu(input_72, inplace=True)
        input_72 = None
        out_b_18 = torch.conv2d(
            out_35,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer2_modules_3_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_19 = torch.nn.functional.batch_norm(
            out_b_18,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_18 = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn2_parameters_bias_ = (None)
        out_b_20 = torch.nn.functional.relu(out_b_19, inplace=True)
        out_b_19 = None
        input_74 = torch._C._nn.avg_pool2d(out_b_20, 4, 4, 0, False, True, None)
        input_75 = torch.conv2d(
            input_74,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_74 = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_76 = torch.nn.functional.batch_norm(
            input_75,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_75 = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_6 = torch.nn.functional.interpolate(input_76, (32, 24))
        input_76 = None
        add_6 = torch.add(out_b_20, interpolate_6)
        interpolate_6 = None
        out_36 = torch.sigmoid(add_6)
        add_6 = None
        input_77 = torch.conv2d(
            out_b_20,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_20 = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_78 = torch.nn.functional.batch_norm(
            input_77,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_77 = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_37 = torch.mul(input_78, out_36)
        input_78 = out_36 = None
        input_79 = torch.conv2d(
            out_37,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_37 = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_80 = torch.nn.functional.batch_norm(
            input_79,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_79 = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_81 = torch.nn.functional.relu(input_80, inplace=True)
        input_80 = None
        cat_6 = torch.cat([input_73, input_81], dim=1)
        input_73 = input_81 = None
        out_38 = torch.conv2d(
            cat_6,
            l_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_6 = l_self_modules_backbone_modules_layer2_modules_3_modules_conv3_parameters_weight_ = (None)
        out_39 = torch.nn.functional.batch_norm(
            out_38,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_38 = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer2_modules_3_modules_bn3_parameters_bias_ = (None)
        out_39 += out_35
        out_40 = out_39
        out_39 = out_35 = None
        out_41 = torch.nn.functional.relu(out_40, inplace=True)
        out_40 = None
        out_a_21 = torch.conv2d(
            out_41,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        out_a_22 = torch.nn.functional.batch_norm(
            out_a_21,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_21 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_ = (None)
        out_a_23 = torch.nn.functional.relu(out_a_22, inplace=True)
        out_a_22 = None
        input_82 = torch.conv2d(
            out_a_23,
            l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_23 = l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_0_parameters_weight_ = (None)
        input_83 = torch.nn.functional.batch_norm(
            input_82,
            l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_82 = l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_k1_modules_1_parameters_bias_ = (None)
        input_84 = torch.nn.functional.relu(input_83, inplace=True)
        input_83 = None
        out_b_21 = torch.conv2d(
            out_41,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_22 = torch.nn.functional.batch_norm(
            out_b_21,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_21 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_ = (None)
        out_b_23 = torch.nn.functional.relu(out_b_22, inplace=True)
        out_b_22 = None
        input_85 = torch._C._nn.avg_pool2d(out_b_23, 4, 4, 0, False, True, None)
        input_86 = torch.conv2d(
            input_85,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_85 = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_87 = torch.nn.functional.batch_norm(
            input_86,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_86 = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_7 = torch.nn.functional.interpolate(input_87, (32, 24))
        input_87 = None
        add_7 = torch.add(out_b_23, interpolate_7)
        interpolate_7 = None
        out_42 = torch.sigmoid(add_7)
        add_7 = None
        input_88 = torch.conv2d(
            out_b_23,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_23 = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_89 = torch.nn.functional.batch_norm(
            input_88,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_88 = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_43 = torch.mul(input_89, out_42)
        input_89 = out_42 = None
        input_90 = torch.conv2d(
            out_43,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        out_43 = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_91 = torch.nn.functional.batch_norm(
            input_90,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_90 = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_92 = torch.nn.functional.relu(input_91, inplace=True)
        input_91 = None
        cat_7 = torch.cat([input_84, input_92], dim=1)
        input_84 = input_92 = None
        out_44 = torch.conv2d(
            cat_7,
            l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_7 = l_self_modules_backbone_modules_layer3_modules_0_modules_conv3_parameters_weight_ = (None)
        out_45 = torch.nn.functional.batch_norm(
            out_44,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_44 = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_bn3_parameters_bias_ = (None)
        input_93 = torch.conv2d(
            out_41,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_41 = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_94 = torch.nn.functional.batch_norm(
            input_93,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_93 = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_45 += input_94
        out_46 = out_45
        out_45 = input_94 = None
        out_47 = torch.nn.functional.relu(out_46, inplace=True)
        out_46 = None
        out_a_24 = torch.conv2d(
            out_47,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        out_a_25 = torch.nn.functional.batch_norm(
            out_a_24,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_24 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_ = (None)
        out_a_26 = torch.nn.functional.relu(out_a_25, inplace=True)
        out_a_25 = None
        input_95 = torch.conv2d(
            out_a_26,
            l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_26 = l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_0_parameters_weight_ = (None)
        input_96 = torch.nn.functional.batch_norm(
            input_95,
            l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_95 = l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_k1_modules_1_parameters_bias_ = (None)
        input_97 = torch.nn.functional.relu(input_96, inplace=True)
        input_96 = None
        out_b_24 = torch.conv2d(
            out_47,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_25 = torch.nn.functional.batch_norm(
            out_b_24,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_24 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_ = (None)
        out_b_26 = torch.nn.functional.relu(out_b_25, inplace=True)
        out_b_25 = None
        input_98 = torch._C._nn.avg_pool2d(out_b_26, 4, 4, 0, False, True, None)
        input_99 = torch.conv2d(
            input_98,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_98 = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_100 = torch.nn.functional.batch_norm(
            input_99,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_99 = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_8 = torch.nn.functional.interpolate(input_100, (16, 12))
        input_100 = None
        add_8 = torch.add(out_b_26, interpolate_8)
        interpolate_8 = None
        out_48 = torch.sigmoid(add_8)
        add_8 = None
        input_101 = torch.conv2d(
            out_b_26,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_26 = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_102 = torch.nn.functional.batch_norm(
            input_101,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_101 = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_49 = torch.mul(input_102, out_48)
        input_102 = out_48 = None
        input_103 = torch.conv2d(
            out_49,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_49 = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_104 = torch.nn.functional.batch_norm(
            input_103,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_103 = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_105 = torch.nn.functional.relu(input_104, inplace=True)
        input_104 = None
        cat_8 = torch.cat([input_97, input_105], dim=1)
        input_97 = input_105 = None
        out_50 = torch.conv2d(
            cat_8,
            l_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_8 = l_self_modules_backbone_modules_layer3_modules_1_modules_conv3_parameters_weight_ = (None)
        out_51 = torch.nn.functional.batch_norm(
            out_50,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_50 = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_1_modules_bn3_parameters_bias_ = (None)
        out_51 += out_47
        out_52 = out_51
        out_51 = out_47 = None
        out_53 = torch.nn.functional.relu(out_52, inplace=True)
        out_52 = None
        out_a_27 = torch.conv2d(
            out_53,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv1_parameters_weight_ = (
            None
        )
        out_a_28 = torch.nn.functional.batch_norm(
            out_a_27,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_27 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn1_parameters_bias_ = (None)
        out_a_29 = torch.nn.functional.relu(out_a_28, inplace=True)
        out_a_28 = None
        input_106 = torch.conv2d(
            out_a_29,
            l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_29 = l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_0_parameters_weight_ = (None)
        input_107 = torch.nn.functional.batch_norm(
            input_106,
            l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_106 = l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_k1_modules_1_parameters_bias_ = (None)
        input_108 = torch.nn.functional.relu(input_107, inplace=True)
        input_107 = None
        out_b_27 = torch.conv2d(
            out_53,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_2_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_28 = torch.nn.functional.batch_norm(
            out_b_27,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_27 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn2_parameters_bias_ = (None)
        out_b_29 = torch.nn.functional.relu(out_b_28, inplace=True)
        out_b_28 = None
        input_109 = torch._C._nn.avg_pool2d(out_b_29, 4, 4, 0, False, True, None)
        input_110 = torch.conv2d(
            input_109,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_109 = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_111 = torch.nn.functional.batch_norm(
            input_110,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_110 = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_9 = torch.nn.functional.interpolate(input_111, (16, 12))
        input_111 = None
        add_9 = torch.add(out_b_29, interpolate_9)
        interpolate_9 = None
        out_54 = torch.sigmoid(add_9)
        add_9 = None
        input_112 = torch.conv2d(
            out_b_29,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_29 = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_113 = torch.nn.functional.batch_norm(
            input_112,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_112 = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_55 = torch.mul(input_113, out_54)
        input_113 = out_54 = None
        input_114 = torch.conv2d(
            out_55,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_55 = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_115 = torch.nn.functional.batch_norm(
            input_114,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_114 = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_116 = torch.nn.functional.relu(input_115, inplace=True)
        input_115 = None
        cat_9 = torch.cat([input_108, input_116], dim=1)
        input_108 = input_116 = None
        out_56 = torch.conv2d(
            cat_9,
            l_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_9 = l_self_modules_backbone_modules_layer3_modules_2_modules_conv3_parameters_weight_ = (None)
        out_57 = torch.nn.functional.batch_norm(
            out_56,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_56 = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_2_modules_bn3_parameters_bias_ = (None)
        out_57 += out_53
        out_58 = out_57
        out_57 = out_53 = None
        out_59 = torch.nn.functional.relu(out_58, inplace=True)
        out_58 = None
        out_a_30 = torch.conv2d(
            out_59,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv1_parameters_weight_ = (
            None
        )
        out_a_31 = torch.nn.functional.batch_norm(
            out_a_30,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_30 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn1_parameters_bias_ = (None)
        out_a_32 = torch.nn.functional.relu(out_a_31, inplace=True)
        out_a_31 = None
        input_117 = torch.conv2d(
            out_a_32,
            l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_32 = l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_0_parameters_weight_ = (None)
        input_118 = torch.nn.functional.batch_norm(
            input_117,
            l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_117 = l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_k1_modules_1_parameters_bias_ = (None)
        input_119 = torch.nn.functional.relu(input_118, inplace=True)
        input_118 = None
        out_b_30 = torch.conv2d(
            out_59,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_3_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_31 = torch.nn.functional.batch_norm(
            out_b_30,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_30 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn2_parameters_bias_ = (None)
        out_b_32 = torch.nn.functional.relu(out_b_31, inplace=True)
        out_b_31 = None
        input_120 = torch._C._nn.avg_pool2d(out_b_32, 4, 4, 0, False, True, None)
        input_121 = torch.conv2d(
            input_120,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_120 = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_122 = torch.nn.functional.batch_norm(
            input_121,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_121 = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_10 = torch.nn.functional.interpolate(input_122, (16, 12))
        input_122 = None
        add_10 = torch.add(out_b_32, interpolate_10)
        interpolate_10 = None
        out_60 = torch.sigmoid(add_10)
        add_10 = None
        input_123 = torch.conv2d(
            out_b_32,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_32 = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_124 = torch.nn.functional.batch_norm(
            input_123,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_123 = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_61 = torch.mul(input_124, out_60)
        input_124 = out_60 = None
        input_125 = torch.conv2d(
            out_61,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_61 = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_126 = torch.nn.functional.batch_norm(
            input_125,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_125 = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_127 = torch.nn.functional.relu(input_126, inplace=True)
        input_126 = None
        cat_10 = torch.cat([input_119, input_127], dim=1)
        input_119 = input_127 = None
        out_62 = torch.conv2d(
            cat_10,
            l_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_10 = l_self_modules_backbone_modules_layer3_modules_3_modules_conv3_parameters_weight_ = (None)
        out_63 = torch.nn.functional.batch_norm(
            out_62,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_62 = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_3_modules_bn3_parameters_bias_ = (None)
        out_63 += out_59
        out_64 = out_63
        out_63 = out_59 = None
        out_65 = torch.nn.functional.relu(out_64, inplace=True)
        out_64 = None
        out_a_33 = torch.conv2d(
            out_65,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv1_parameters_weight_ = (
            None
        )
        out_a_34 = torch.nn.functional.batch_norm(
            out_a_33,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_33 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn1_parameters_bias_ = (None)
        out_a_35 = torch.nn.functional.relu(out_a_34, inplace=True)
        out_a_34 = None
        input_128 = torch.conv2d(
            out_a_35,
            l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_35 = l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_0_parameters_weight_ = (None)
        input_129 = torch.nn.functional.batch_norm(
            input_128,
            l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_128 = l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_k1_modules_1_parameters_bias_ = (None)
        input_130 = torch.nn.functional.relu(input_129, inplace=True)
        input_129 = None
        out_b_33 = torch.conv2d(
            out_65,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_4_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_34 = torch.nn.functional.batch_norm(
            out_b_33,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_33 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn2_parameters_bias_ = (None)
        out_b_35 = torch.nn.functional.relu(out_b_34, inplace=True)
        out_b_34 = None
        input_131 = torch._C._nn.avg_pool2d(out_b_35, 4, 4, 0, False, True, None)
        input_132 = torch.conv2d(
            input_131,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_131 = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_133 = torch.nn.functional.batch_norm(
            input_132,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_132 = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_11 = torch.nn.functional.interpolate(input_133, (16, 12))
        input_133 = None
        add_11 = torch.add(out_b_35, interpolate_11)
        interpolate_11 = None
        out_66 = torch.sigmoid(add_11)
        add_11 = None
        input_134 = torch.conv2d(
            out_b_35,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_35 = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_135 = torch.nn.functional.batch_norm(
            input_134,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_134 = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_67 = torch.mul(input_135, out_66)
        input_135 = out_66 = None
        input_136 = torch.conv2d(
            out_67,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_67 = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_137 = torch.nn.functional.batch_norm(
            input_136,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_136 = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_138 = torch.nn.functional.relu(input_137, inplace=True)
        input_137 = None
        cat_11 = torch.cat([input_130, input_138], dim=1)
        input_130 = input_138 = None
        out_68 = torch.conv2d(
            cat_11,
            l_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_11 = l_self_modules_backbone_modules_layer3_modules_4_modules_conv3_parameters_weight_ = (None)
        out_69 = torch.nn.functional.batch_norm(
            out_68,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_68 = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_4_modules_bn3_parameters_bias_ = (None)
        out_69 += out_65
        out_70 = out_69
        out_69 = out_65 = None
        out_71 = torch.nn.functional.relu(out_70, inplace=True)
        out_70 = None
        out_a_36 = torch.conv2d(
            out_71,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv1_parameters_weight_ = (
            None
        )
        out_a_37 = torch.nn.functional.batch_norm(
            out_a_36,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_36 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn1_parameters_bias_ = (None)
        out_a_38 = torch.nn.functional.relu(out_a_37, inplace=True)
        out_a_37 = None
        input_139 = torch.conv2d(
            out_a_38,
            l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_38 = l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_0_parameters_weight_ = (None)
        input_140 = torch.nn.functional.batch_norm(
            input_139,
            l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_139 = l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_k1_modules_1_parameters_bias_ = (None)
        input_141 = torch.nn.functional.relu(input_140, inplace=True)
        input_140 = None
        out_b_36 = torch.conv2d(
            out_71,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer3_modules_5_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_37 = torch.nn.functional.batch_norm(
            out_b_36,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_36 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn2_parameters_bias_ = (None)
        out_b_38 = torch.nn.functional.relu(out_b_37, inplace=True)
        out_b_37 = None
        input_142 = torch._C._nn.avg_pool2d(out_b_38, 4, 4, 0, False, True, None)
        input_143 = torch.conv2d(
            input_142,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_142 = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_144 = torch.nn.functional.batch_norm(
            input_143,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_143 = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_12 = torch.nn.functional.interpolate(input_144, (16, 12))
        input_144 = None
        add_12 = torch.add(out_b_38, interpolate_12)
        interpolate_12 = None
        out_72 = torch.sigmoid(add_12)
        add_12 = None
        input_145 = torch.conv2d(
            out_b_38,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_38 = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_146 = torch.nn.functional.batch_norm(
            input_145,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_145 = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_73 = torch.mul(input_146, out_72)
        input_146 = out_72 = None
        input_147 = torch.conv2d(
            out_73,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_73 = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_148 = torch.nn.functional.batch_norm(
            input_147,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_147 = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_149 = torch.nn.functional.relu(input_148, inplace=True)
        input_148 = None
        cat_12 = torch.cat([input_141, input_149], dim=1)
        input_141 = input_149 = None
        out_74 = torch.conv2d(
            cat_12,
            l_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_12 = l_self_modules_backbone_modules_layer3_modules_5_modules_conv3_parameters_weight_ = (None)
        out_75 = torch.nn.functional.batch_norm(
            out_74,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_74 = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer3_modules_5_modules_bn3_parameters_bias_ = (None)
        out_75 += out_71
        out_76 = out_75
        out_75 = out_71 = None
        out_77 = torch.nn.functional.relu(out_76, inplace=True)
        out_76 = None
        out_a_39 = torch.conv2d(
            out_77,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_ = (
            None
        )
        out_a_40 = torch.nn.functional.batch_norm(
            out_a_39,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_39 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_ = (None)
        out_a_41 = torch.nn.functional.relu(out_a_40, inplace=True)
        out_a_40 = None
        input_150 = torch.conv2d(
            out_a_41,
            l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_41 = l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_0_parameters_weight_ = (None)
        input_151 = torch.nn.functional.batch_norm(
            input_150,
            l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_150 = l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_k1_modules_1_parameters_bias_ = (None)
        input_152 = torch.nn.functional.relu(input_151, inplace=True)
        input_151 = None
        out_b_39 = torch.conv2d(
            out_77,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_40 = torch.nn.functional.batch_norm(
            out_b_39,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_39 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_ = (None)
        out_b_41 = torch.nn.functional.relu(out_b_40, inplace=True)
        out_b_40 = None
        input_153 = torch._C._nn.avg_pool2d(out_b_41, 4, 4, 0, False, True, None)
        input_154 = torch.conv2d(
            input_153,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_153 = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_155 = torch.nn.functional.batch_norm(
            input_154,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_154 = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_13 = torch.nn.functional.interpolate(input_155, (16, 12))
        input_155 = None
        add_13 = torch.add(out_b_41, interpolate_13)
        interpolate_13 = None
        out_78 = torch.sigmoid(add_13)
        add_13 = None
        input_156 = torch.conv2d(
            out_b_41,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_41 = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_157 = torch.nn.functional.batch_norm(
            input_156,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_156 = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_79 = torch.mul(input_157, out_78)
        input_157 = out_78 = None
        input_158 = torch.conv2d(
            out_79,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        out_79 = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_159 = torch.nn.functional.batch_norm(
            input_158,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_158 = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_160 = torch.nn.functional.relu(input_159, inplace=True)
        input_159 = None
        cat_13 = torch.cat([input_152, input_160], dim=1)
        input_152 = input_160 = None
        out_80 = torch.conv2d(
            cat_13,
            l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_13 = l_self_modules_backbone_modules_layer4_modules_0_modules_conv3_parameters_weight_ = (None)
        out_81 = torch.nn.functional.batch_norm(
            out_80,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_80 = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_bn3_parameters_bias_ = (None)
        input_161 = torch.conv2d(
            out_77,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_77 = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_162 = torch.nn.functional.batch_norm(
            input_161,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_161 = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        out_81 += input_162
        out_82 = out_81
        out_81 = input_162 = None
        out_83 = torch.nn.functional.relu(out_82, inplace=True)
        out_82 = None
        out_a_42 = torch.conv2d(
            out_83,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_ = (
            None
        )
        out_a_43 = torch.nn.functional.batch_norm(
            out_a_42,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_42 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_ = (None)
        out_a_44 = torch.nn.functional.relu(out_a_43, inplace=True)
        out_a_43 = None
        input_163 = torch.conv2d(
            out_a_44,
            l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_44 = l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_0_parameters_weight_ = (None)
        input_164 = torch.nn.functional.batch_norm(
            input_163,
            l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_163 = l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_k1_modules_1_parameters_bias_ = (None)
        input_165 = torch.nn.functional.relu(input_164, inplace=True)
        input_164 = None
        out_b_42 = torch.conv2d(
            out_83,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_43 = torch.nn.functional.batch_norm(
            out_b_42,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_42 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_ = (None)
        out_b_44 = torch.nn.functional.relu(out_b_43, inplace=True)
        out_b_43 = None
        input_166 = torch._C._nn.avg_pool2d(out_b_44, 4, 4, 0, False, True, None)
        input_167 = torch.conv2d(
            input_166,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_166 = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_168 = torch.nn.functional.batch_norm(
            input_167,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_167 = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_14 = torch.nn.functional.interpolate(input_168, (8, 6))
        input_168 = None
        add_14 = torch.add(out_b_44, interpolate_14)
        interpolate_14 = None
        out_84 = torch.sigmoid(add_14)
        add_14 = None
        input_169 = torch.conv2d(
            out_b_44,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_44 = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_170 = torch.nn.functional.batch_norm(
            input_169,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_169 = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_85 = torch.mul(input_170, out_84)
        input_170 = out_84 = None
        input_171 = torch.conv2d(
            out_85,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_85 = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_172 = torch.nn.functional.batch_norm(
            input_171,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_171 = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_173 = torch.nn.functional.relu(input_172, inplace=True)
        input_172 = None
        cat_14 = torch.cat([input_165, input_173], dim=1)
        input_165 = input_173 = None
        out_86 = torch.conv2d(
            cat_14,
            l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_14 = l_self_modules_backbone_modules_layer4_modules_1_modules_conv3_parameters_weight_ = (None)
        out_87 = torch.nn.functional.batch_norm(
            out_86,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_86 = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_1_modules_bn3_parameters_bias_ = (None)
        out_87 += out_83
        out_88 = out_87
        out_87 = out_83 = None
        out_89 = torch.nn.functional.relu(out_88, inplace=True)
        out_88 = None
        out_a_45 = torch.conv2d(
            out_89,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv1_parameters_weight_ = (
            None
        )
        out_a_46 = torch.nn.functional.batch_norm(
            out_a_45,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_a_45 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn1_parameters_bias_ = (None)
        out_a_47 = torch.nn.functional.relu(out_a_46, inplace=True)
        out_a_46 = None
        input_174 = torch.conv2d(
            out_a_47,
            l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_a_47 = l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_0_parameters_weight_ = (None)
        input_175 = torch.nn.functional.batch_norm(
            input_174,
            l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_174 = l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_k1_modules_1_parameters_bias_ = (None)
        input_176 = torch.nn.functional.relu(input_175, inplace=True)
        input_175 = None
        out_b_45 = torch.conv2d(
            out_89,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_layer4_modules_2_modules_conv2_parameters_weight_ = (
            None
        )
        out_b_46 = torch.nn.functional.batch_norm(
            out_b_45,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_b_45 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn2_parameters_bias_ = (None)
        out_b_47 = torch.nn.functional.relu(out_b_46, inplace=True)
        out_b_46 = None
        input_177 = torch._C._nn.avg_pool2d(out_b_47, 4, 4, 0, False, True, None)
        input_178 = torch.conv2d(
            input_177,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_177 = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_1_parameters_weight_ = (None)
        input_179 = torch.nn.functional.batch_norm(
            input_178,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_178 = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k2_modules_2_parameters_bias_ = (None)
        interpolate_15 = torch.nn.functional.interpolate(input_179, (8, 6))
        input_179 = None
        add_15 = torch.add(out_b_47, interpolate_15)
        interpolate_15 = None
        out_90 = torch.sigmoid(add_15)
        add_15 = None
        input_180 = torch.conv2d(
            out_b_47,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_b_47 = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_0_parameters_weight_ = (None)
        input_181 = torch.nn.functional.batch_norm(
            input_180,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_180 = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k3_modules_1_parameters_bias_ = (None)
        out_91 = torch.mul(input_181, out_90)
        input_181 = out_90 = None
        input_182 = torch.conv2d(
            out_91,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        out_91 = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_0_parameters_weight_ = (None)
        input_183 = torch.nn.functional.batch_norm(
            input_182,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_182 = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_scconv_modules_k4_modules_1_parameters_bias_ = (None)
        input_184 = torch.nn.functional.relu(input_183, inplace=True)
        input_183 = None
        cat_15 = torch.cat([input_176, input_184], dim=1)
        input_176 = input_184 = None
        out_92 = torch.conv2d(
            cat_15,
            l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_15 = l_self_modules_backbone_modules_layer4_modules_2_modules_conv3_parameters_weight_ = (None)
        out_93 = torch.nn.functional.batch_norm(
            out_92,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_92 = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_mean_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_buffers_running_var_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_weight_ = l_self_modules_backbone_modules_layer4_modules_2_modules_bn3_parameters_bias_ = (None)
        out_93 += out_89
        out_94 = out_93
        out_93 = out_89 = None
        out_95 = torch.nn.functional.relu(out_94, inplace=True)
        out_94 = None
        input_185 = torch.conv_transpose2d(
            out_95,
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        out_95 = (
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_
        ) = None
        input_186 = torch.nn.functional.batch_norm(
            input_185,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_185 = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_ = None
        input_187 = torch.nn.functional.relu(input_186, inplace=True)
        input_186 = None
        input_188 = torch.conv_transpose2d(
            input_187,
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        input_187 = (
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_
        ) = None
        input_189 = torch.nn.functional.batch_norm(
            input_188,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_188 = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_ = None
        input_190 = torch.nn.functional.relu(input_189, inplace=True)
        input_189 = None
        input_191 = torch.conv_transpose2d(
            input_190,
            l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        input_190 = (
            l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_
        ) = None
        input_192 = torch.nn.functional.batch_norm(
            input_191,
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_191 = (
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_ = None
        input_193 = torch.nn.functional.relu(input_192, inplace=True)
        input_192 = None
        x_4 = torch.conv2d(
            input_193,
            l_self_modules_head_modules_final_layer_parameters_weight_,
            l_self_modules_head_modules_final_layer_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_193 = (
            l_self_modules_head_modules_final_layer_parameters_weight_
        ) = l_self_modules_head_modules_final_layer_parameters_bias_ = None
        return (x_4,)
