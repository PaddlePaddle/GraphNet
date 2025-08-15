import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_compression_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_compression_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_compression_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_down_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_down_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_down_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_down_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_down_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_compression_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_compression_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_compression_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_down_2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_down_2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_down_2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_down_2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_down_2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_down_2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_down_2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_down_2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_down_2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_down_2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_compression_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_compression_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_compression_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_compression_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_compression_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_shortcut_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_head_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_head_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_head_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_head_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_head_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_head_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_head_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_head_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_head_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_ = L_inputs_
        l_self_modules_backbone_modules_stem_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_stem_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_compression_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_compression_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_compression_1_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_compression_1_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_compression_1_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_compression_1_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_down_1_modules_conv_parameters_weight_ = (
            L_self_modules_backbone_modules_down_1_modules_conv_parameters_weight_
        )
        l_self_modules_backbone_modules_down_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_backbone_modules_down_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_backbone_modules_down_1_modules_bn_buffers_running_var_ = (
            L_self_modules_backbone_modules_down_1_modules_bn_buffers_running_var_
        )
        l_self_modules_backbone_modules_down_1_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_down_1_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_down_1_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_down_1_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_compression_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_compression_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_compression_2_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_compression_2_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_compression_2_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_compression_2_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_down_2_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_down_2_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_down_2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_down_2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_down_2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_down_2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_down_2_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_down_2_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_down_2_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_down_2_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_down_2_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_down_2_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_down_2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_down_2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_down_2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_down_2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_down_2_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_down_2_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_down_2_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_down_2_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_compression_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spp_modules_compression_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spp_modules_compression_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spp_modules_compression_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spp_modules_compression_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_compression_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_compression_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spp_modules_compression_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spp_modules_compression_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_compression_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spp_modules_shortcut_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_shortcut_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_head_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_head_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_head_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_head_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_head_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_head_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_head_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_head_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_head_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_head_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_head_modules_1_buffers_running_mean_ = (
            L_self_modules_decode_head_modules_head_modules_1_buffers_running_mean_
        )
        l_self_modules_decode_head_modules_head_modules_1_buffers_running_var_ = (
            L_self_modules_decode_head_modules_head_modules_1_buffers_running_var_
        )
        l_self_modules_decode_head_modules_head_modules_1_parameters_weight_ = (
            L_self_modules_decode_head_modules_head_modules_1_parameters_weight_
        )
        l_self_modules_decode_head_modules_head_modules_1_parameters_bias_ = (
            L_self_modules_decode_head_modules_head_modules_1_parameters_bias_
        )
        l_self_modules_decode_head_modules_conv_seg_parameters_weight_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_weight_
        )
        l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_bias_
        )
        x = torch.conv2d(
            l_inputs_,
            l_self_modules_backbone_modules_stem_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_inputs_ = l_self_modules_backbone_modules_stem_modules_0_modules_conv_parameters_weight_ = (None)
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_bias_
        ) = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_backbone_modules_stem_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_backbone_modules_stem_modules_1_modules_conv_parameters_weight_ = (None)
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = l_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_bias_
        ) = None
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_8 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_8 = l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_10 += x_5
        out = x_10
        x_10 = x_5 = None
        out_1 = torch.nn.functional.relu(out, inplace=True)
        out = None
        x_11 = torch.conv2d(
            out_1,
            l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_12 = torch.nn.functional.batch_norm(
            x_11,
            l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_11 = l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_13 = torch.nn.functional.relu(x_12, inplace=True)
        x_12 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_14 = l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_2_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_15 += out_1
        out_2 = x_15
        x_15 = out_1 = None
        input_1 = torch.nn.functional.relu(out_2, inplace=False)
        out_2 = None
        x_16 = torch.conv2d(
            input_1,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_18 = torch.nn.functional.relu(x_17, inplace=True)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        input_2 = torch.conv2d(
            input_1,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        input_1 = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_3 = torch.nn.functional.batch_norm(
            input_2,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_2 = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_20 += input_3
        out_3 = x_20
        x_20 = input_3 = None
        out_4 = torch.nn.functional.relu(out_3, inplace=True)
        out_3 = None
        x_21 = torch.conv2d(
            out_4,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_21 = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_23 = torch.nn.functional.relu(x_22, inplace=True)
        x_22 = None
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_23 = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_24 = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_25 += out_4
        out_5 = x_25
        x_25 = out_4 = None
        input_4 = torch.nn.functional.relu(out_5, inplace=False)
        out_5 = None
        x_26 = torch.conv2d(
            input_4,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_28 = torch.nn.functional.relu(x_27, inplace=True)
        x_27 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        input_5 = torch.conv2d(
            input_4,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_0_parameters_weight_ = (
            None
        )
        input_6 = torch.nn.functional.batch_norm(
            input_5,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_5 = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_30 += input_6
        out_6 = x_30
        x_30 = input_6 = None
        out_7 = torch.nn.functional.relu(out_6, inplace=True)
        out_6 = None
        x_31 = torch.conv2d(
            out_7,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_33 = torch.nn.functional.relu(x_32, inplace=True)
        x_32 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_33 = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_context_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_35 += out_7
        out_8 = x_35
        x_35 = out_7 = None
        x_36 = torch.conv2d(
            input_4,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_38 = torch.nn.functional.relu(x_37, inplace=True)
        x_37 = None
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_38 = l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_39 = l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_40 += input_4
        out_9 = x_40
        x_40 = input_4 = None
        out_10 = torch.nn.functional.relu(out_9, inplace=True)
        out_9 = None
        x_41 = torch.conv2d(
            out_10,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_42 = torch.nn.functional.batch_norm(
            x_41,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_41 = l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_43 = torch.nn.functional.relu(x_42, inplace=True)
        x_42 = None
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_43 = l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_45 = torch.nn.functional.batch_norm(
            x_44,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_44 = l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_45 += out_10
        out_11 = x_45
        x_45 = out_10 = None
        relu_16 = torch.nn.functional.relu(out_8, inplace=False)
        x_46 = torch.conv2d(
            relu_16,
            l_self_modules_backbone_modules_compression_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        relu_16 = l_self_modules_backbone_modules_compression_1_modules_conv_parameters_weight_ = (None)
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_compression_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_compression_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_46 = l_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_compression_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_compression_1_modules_bn_parameters_bias_
        ) = None
        relu_17 = torch.nn.functional.relu(out_11, inplace=False)
        x_48 = torch.conv2d(
            relu_17,
            l_self_modules_backbone_modules_down_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        relu_17 = (
            l_self_modules_backbone_modules_down_1_modules_conv_parameters_weight_
        ) = None
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_backbone_modules_down_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_down_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_down_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_down_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_48 = (
            l_self_modules_backbone_modules_down_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_down_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_down_1_modules_bn_parameters_weight_
        ) = l_self_modules_backbone_modules_down_1_modules_bn_parameters_bias_ = None
        out_8 += x_49
        x_c = out_8
        out_8 = x_49 = None
        interpolate = torch.nn.functional.interpolate(
            x_47, (64, 64), None, "bilinear", False
        )
        x_47 = None
        out_11 += interpolate
        x_s = out_11
        out_11 = interpolate = None
        residual = torch.nn.functional.relu(x_c, inplace=False)
        x_c = None
        x_50 = torch.conv2d(
            residual,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        input_7 = torch.conv2d(
            residual,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        residual = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_54 += input_8
        out_12 = x_54
        x_54 = input_8 = None
        out_13 = torch.nn.functional.relu(out_12, inplace=True)
        out_12 = None
        x_55 = torch.conv2d(
            out_13,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_55 = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_57 = torch.nn.functional.relu(x_56, inplace=True)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_context_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_59 += out_13
        out_14 = x_59
        x_59 = out_13 = None
        residual_1 = torch.nn.functional.relu(x_s, inplace=False)
        x_s = None
        x_60 = torch.conv2d(
            residual_1,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_60 = l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_62 = torch.nn.functional.relu(x_61, inplace=True)
        x_61 = None
        x_63 = torch.conv2d(
            x_62,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_62 = l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_64 += residual_1
        out_15 = x_64
        x_64 = residual_1 = None
        out_16 = torch.nn.functional.relu(out_15, inplace=True)
        out_15 = None
        x_65 = torch.conv2d(
            out_16,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_65 = l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_67 = torch.nn.functional.relu(x_66, inplace=True)
        x_66 = None
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_69 += out_16
        out_17 = x_69
        x_69 = out_16 = None
        relu_26 = torch.nn.functional.relu(out_14, inplace=False)
        x_70 = torch.conv2d(
            relu_26,
            l_self_modules_backbone_modules_compression_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        relu_26 = l_self_modules_backbone_modules_compression_2_modules_conv_parameters_weight_ = (None)
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_compression_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_compression_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = l_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_compression_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_compression_2_modules_bn_parameters_bias_
        ) = None
        relu_27 = torch.nn.functional.relu(out_17, inplace=False)
        x_72 = torch.conv2d(
            relu_27,
            l_self_modules_backbone_modules_down_2_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        relu_27 = l_self_modules_backbone_modules_down_2_modules_0_modules_conv_parameters_weight_ = (None)
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_backbone_modules_down_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_down_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_down_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_down_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = l_self_modules_backbone_modules_down_2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_down_2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_down_2_modules_0_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_down_2_modules_0_modules_bn_parameters_bias_
        ) = None
        x_74 = torch.nn.functional.relu(x_73, inplace=True)
        x_73 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_backbone_modules_down_2_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_backbone_modules_down_2_modules_1_modules_conv_parameters_weight_ = (None)
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_backbone_modules_down_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_down_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_down_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_down_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = l_self_modules_backbone_modules_down_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_down_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_down_2_modules_1_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_down_2_modules_1_modules_bn_parameters_bias_
        ) = None
        out_14 += x_76
        x_c_1 = out_14
        out_14 = x_76 = None
        interpolate_1 = torch.nn.functional.interpolate(
            x_71, (64, 64), None, "bilinear", False
        )
        x_71 = None
        out_17 += interpolate_1
        x_s_1 = out_17
        out_17 = interpolate_1 = None
        residual_2 = torch.nn.functional.relu(x_s_1, inplace=False)
        x_s_1 = None
        x_77 = torch.conv2d(
            residual_2,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_77 = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_79 = torch.nn.functional.relu(x_78, inplace=True)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_82 = torch.nn.functional.relu(x_81, inplace=True)
        x_81 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_ = (None)
        input_9 = torch.conv2d(
            residual_2,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        residual_2 = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_10 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_9 = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_spatial_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_84 += input_10
        out_18 = x_84
        x_84 = input_10 = None
        residual_3 = torch.nn.functional.relu(x_c_1, inplace=False)
        x_c_1 = None
        x_85 = torch.conv2d(
            residual_3,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_85 = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_87 = torch.nn.functional.relu(x_86, inplace=True)
        x_86 = None
        x_88 = torch.conv2d(
            x_87,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_87 = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_90 = torch.nn.functional.relu(x_89, inplace=True)
        x_89 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_90 = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_ = (None)
        input_11 = torch.conv2d(
            residual_3,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        residual_3 = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_12 = torch.nn.functional.batch_norm(
            input_11,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_11 = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_backbone_modules_context_branch_layers_modules_2_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_92 += input_12
        out_19 = x_92
        x_92 = input_12 = None
        x_93 = torch.nn.functional.batch_norm(
            out_19,
            l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_parameters_bias_ = (None)
        x_94 = torch.nn.functional.relu(x_93, inplace=True)
        x_93 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_94 = l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_conv_parameters_weight_ = (None)
        input_13 = torch._C._nn.avg_pool2d(out_19, 5, 2, 2, False, True, None)
        x_96 = torch.nn.functional.batch_norm(
            input_13,
            l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_13 = l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        x_97 = torch.nn.functional.relu(x_96, inplace=True)
        x_96 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        feat_up = torch.nn.functional.interpolate(x_98, size=(8, 8), mode="bilinear")
        x_98 = None
        add = feat_up + x_95
        feat_up = None
        x_99 = torch.nn.functional.batch_norm(
            add,
            l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        add = l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_parameters_bias_ = (None)
        x_100 = torch.nn.functional.relu(x_99, inplace=True)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_100 = l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_conv_parameters_weight_ = (None)
        input_14 = torch._C._nn.avg_pool2d(out_19, 9, 4, 4, False, True, None)
        x_102 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_14 = l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        x_103 = torch.nn.functional.relu(x_102, inplace=True)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        feat_up_1 = torch.nn.functional.interpolate(x_104, size=(8, 8), mode="bilinear")
        x_104 = None
        add_1 = feat_up_1 + x_101
        feat_up_1 = None
        x_105 = torch.nn.functional.batch_norm(
            add_1,
            l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        add_1 = l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_parameters_bias_ = (None)
        x_106 = torch.nn.functional.relu(x_105, inplace=True)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_106 = l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_conv_parameters_weight_ = (None)
        input_15 = torch._C._nn.avg_pool2d(out_19, 17, 8, 8, False, True, None)
        x_108 = torch.nn.functional.batch_norm(
            input_15,
            l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_15 = l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_bias_ = (None)
        x_109 = torch.nn.functional.relu(x_108, inplace=True)
        x_108 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_conv_parameters_weight_ = (None)
        feat_up_2 = torch.nn.functional.interpolate(x_110, size=(8, 8), mode="bilinear")
        x_110 = None
        add_2 = feat_up_2 + x_107
        feat_up_2 = None
        x_111 = torch.nn.functional.batch_norm(
            add_2,
            l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        add_2 = l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_parameters_bias_ = (None)
        x_112 = torch.nn.functional.relu(x_111, inplace=True)
        x_111 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_112 = l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_conv_parameters_weight_ = (None)
        input_16 = torch.nn.functional.adaptive_avg_pool2d(out_19, (1, 1))
        x_114 = torch.nn.functional.batch_norm(
            input_16,
            l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_16 = l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_bias_ = (None)
        x_115 = torch.nn.functional.relu(x_114, inplace=True)
        x_114 = None
        x_116 = torch.conv2d(
            x_115,
            l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_115 = l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_conv_parameters_weight_ = (None)
        feat_up_3 = torch.nn.functional.interpolate(x_116, size=(8, 8), mode="bilinear")
        x_116 = None
        add_3 = feat_up_3 + x_113
        feat_up_3 = None
        x_117 = torch.nn.functional.batch_norm(
            add_3,
            l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        add_3 = l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_parameters_bias_ = (None)
        x_118 = torch.nn.functional.relu(x_117, inplace=True)
        x_117 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_conv_parameters_weight_ = (None)
        cat = torch.cat([x_95, x_101, x_107, x_113, x_119], dim=1)
        x_95 = x_101 = x_107 = x_113 = x_119 = None
        x_120 = torch.nn.functional.batch_norm(
            cat,
            l_self_modules_backbone_modules_spp_modules_compression_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_compression_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_compression_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_compression_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        cat = l_self_modules_backbone_modules_spp_modules_compression_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_compression_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_compression_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_compression_modules_bn_parameters_bias_ = (None)
        x_121 = torch.nn.functional.relu(x_120, inplace=True)
        x_120 = None
        x_122 = torch.conv2d(
            x_121,
            l_self_modules_backbone_modules_spp_modules_compression_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_121 = l_self_modules_backbone_modules_spp_modules_compression_modules_conv_parameters_weight_ = (None)
        x_123 = torch.nn.functional.batch_norm(
            out_19,
            l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_19 = l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_parameters_bias_ = (None)
        x_124 = torch.nn.functional.relu(x_123, inplace=True)
        x_123 = None
        x_125 = torch.conv2d(
            x_124,
            l_self_modules_backbone_modules_spp_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_124 = l_self_modules_backbone_modules_spp_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_c_2 = x_122 + x_125
        x_122 = x_125 = None
        x_c_3 = torch.nn.functional.interpolate(
            x_c_2, (64, 64), None, "bilinear", False
        )
        x_c_2 = None
        x_126 = out_18 + x_c_3
        out_18 = x_c_3 = None
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_decode_head_modules_head_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_head_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_head_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_head_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = l_self_modules_decode_head_modules_head_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_head_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_head_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_head_modules_0_modules_bn_parameters_bias_ = (None)
        x_128 = torch.nn.functional.relu(x_127, inplace=True)
        x_127 = None
        x_129 = torch.conv2d(
            x_128,
            l_self_modules_decode_head_modules_head_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_128 = l_self_modules_decode_head_modules_head_modules_0_modules_conv_parameters_weight_ = (None)
        input_17 = torch.nn.functional.batch_norm(
            x_129,
            l_self_modules_decode_head_modules_head_modules_1_buffers_running_mean_,
            l_self_modules_decode_head_modules_head_modules_1_buffers_running_var_,
            l_self_modules_decode_head_modules_head_modules_1_parameters_weight_,
            l_self_modules_decode_head_modules_head_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_129 = (
            l_self_modules_decode_head_modules_head_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_decode_head_modules_head_modules_1_buffers_running_var_
        ) = (
            l_self_modules_decode_head_modules_head_modules_1_parameters_weight_
        ) = l_self_modules_decode_head_modules_head_modules_1_parameters_bias_ = None
        input_18 = torch.nn.functional.relu(input_17, inplace=True)
        input_17 = None
        output = torch.conv2d(
            input_18,
            l_self_modules_decode_head_modules_conv_seg_parameters_weight_,
            l_self_modules_decode_head_modules_conv_seg_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_18 = (
            l_self_modules_decode_head_modules_conv_seg_parameters_weight_
        ) = l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = None
        return (output,)
