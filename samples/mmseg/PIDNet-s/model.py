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
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_compression_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_compression_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_compression_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_pag_1_modules_f_i_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_pag_1_modules_f_p_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_diff_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_diff_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_diff_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_diff_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_diff_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_compression_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_compression_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_compression_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_pag_2_modules_f_i_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_pag_2_modules_f_p_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_diff_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_diff_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_diff_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_diff_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_diff_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_processes_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_processes_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_spp_modules_processes_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_processes_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_spp_modules_processes_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_dfm_modules_f_p_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_dfm_modules_f_i_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_i_head_modules_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_i_head_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_i_head_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_i_head_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_i_head_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_parameters_bias_
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
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_compression_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_compression_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_compression_1_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_compression_1_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_compression_1_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_compression_1_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_pag_1_modules_f_i_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_pag_1_modules_f_i_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_pag_1_modules_f_p_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_pag_1_modules_f_p_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_diff_1_modules_conv_parameters_weight_ = (
            L_self_modules_backbone_modules_diff_1_modules_conv_parameters_weight_
        )
        l_self_modules_backbone_modules_diff_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_backbone_modules_diff_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_backbone_modules_diff_1_modules_bn_buffers_running_var_ = (
            L_self_modules_backbone_modules_diff_1_modules_bn_buffers_running_var_
        )
        l_self_modules_backbone_modules_diff_1_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_diff_1_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_diff_1_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_diff_1_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_compression_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_compression_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_compression_2_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_compression_2_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_compression_2_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_compression_2_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_pag_2_modules_f_i_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_pag_2_modules_f_i_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_pag_2_modules_f_p_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_pag_2_modules_f_p_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_diff_2_modules_conv_parameters_weight_ = (
            L_self_modules_backbone_modules_diff_2_modules_conv_parameters_weight_
        )
        l_self_modules_backbone_modules_diff_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_backbone_modules_diff_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_backbone_modules_diff_2_modules_bn_buffers_running_var_ = (
            L_self_modules_backbone_modules_diff_2_modules_bn_buffers_running_var_
        )
        l_self_modules_backbone_modules_diff_2_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_diff_2_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_diff_2_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_diff_2_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_
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
        l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_processes_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_spp_modules_processes_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_spp_modules_processes_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_spp_modules_processes_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_spp_modules_processes_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_processes_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_spp_modules_processes_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_spp_modules_processes_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_spp_modules_processes_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_spp_modules_processes_modules_conv_parameters_weight_
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
        l_self_modules_backbone_modules_dfm_modules_f_p_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_dfm_modules_f_p_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_dfm_modules_f_i_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_dfm_modules_f_i_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_parameters_bias_
        )
        l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_i_head_modules_conv_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_i_head_modules_conv_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_i_head_modules_norm_buffers_running_mean_ = (
            L_self_modules_decode_head_modules_i_head_modules_norm_buffers_running_mean_
        )
        l_self_modules_decode_head_modules_i_head_modules_norm_buffers_running_var_ = (
            L_self_modules_decode_head_modules_i_head_modules_norm_buffers_running_var_
        )
        l_self_modules_decode_head_modules_i_head_modules_norm_parameters_weight_ = (
            L_self_modules_decode_head_modules_i_head_modules_norm_parameters_weight_
        )
        l_self_modules_decode_head_modules_i_head_modules_norm_parameters_bias_ = (
            L_self_modules_decode_head_modules_i_head_modules_norm_parameters_bias_
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
        x_21 = torch.conv2d(
            input_1,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        input_1 = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_21 = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_20 += x_22
        out_3 = x_20
        x_20 = x_22 = None
        out_4 = torch.nn.functional.relu(out_3, inplace=True)
        out_3 = None
        x_23 = torch.conv2d(
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
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_25 = torch.nn.functional.relu(x_24, inplace=True)
        x_24 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_27 += out_4
        out_5 = x_27
        x_27 = out_4 = None
        input_2 = torch.nn.functional.relu(out_5, inplace=False)
        out_5 = None
        x_28 = torch.conv2d(
            input_2,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_30 = torch.nn.functional.relu(x_29, inplace=True)
        x_29 = None
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_30 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_33 = torch.conv2d(
            input_2,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_conv_parameters_weight_ = (
            None
        )
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_32 += x_34
        out_6 = x_32
        x_32 = x_34 = None
        out_7 = torch.nn.functional.relu(out_6, inplace=True)
        out_6 = None
        x_35 = torch.conv2d(
            out_7,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_37 = torch.nn.functional.relu(x_36, inplace=True)
        x_36 = None
        x_38 = torch.conv2d(
            x_37,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_39 += out_7
        out_8 = x_39
        x_39 = out_7 = None
        out_9 = torch.nn.functional.relu(out_8, inplace=True)
        out_8 = None
        x_40 = torch.conv2d(
            out_9,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_42 = torch.nn.functional.relu(x_41, inplace=True)
        x_41 = None
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_42 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_44 += out_9
        out_10 = x_44
        x_44 = out_9 = None
        x_i = torch.nn.functional.relu(out_10, inplace=False)
        out_10 = None
        x_45 = torch.conv2d(
            input_2,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_47 = torch.nn.functional.relu(x_46, inplace=True)
        x_46 = None
        x_48 = torch.conv2d(
            x_47,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_47 = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_48 = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_49 += input_2
        out_11 = x_49
        x_49 = None
        out_12 = torch.nn.functional.relu(out_11, inplace=True)
        out_11 = None
        x_50 = torch.conv2d(
            out_12,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_54 += out_12
        out_13 = x_54
        x_54 = out_12 = None
        x_55 = torch.conv2d(
            input_2,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_55 = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_57 = torch.nn.functional.relu(x_56, inplace=True)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_60 = torch.conv2d(
            input_2,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_2 = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_60 = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_59 += x_61
        out_14 = x_59
        x_59 = x_61 = None
        x_62 = torch.conv2d(
            x_i,
            l_self_modules_backbone_modules_compression_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_compression_1_modules_conv_parameters_weight_ = (
            None
        )
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_compression_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_compression_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_62 = l_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_compression_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_compression_1_modules_bn_parameters_bias_
        ) = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_backbone_modules_pag_1_modules_f_i_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_pag_1_modules_f_i_modules_conv_parameters_weight_ = (
            None
        )
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_parameters_bias_ = (None)
        f_i = torch.nn.functional.interpolate(
            x_65, size=(64, 64), mode="bilinear", align_corners=False
        )
        x_65 = None
        x_66 = torch.conv2d(
            out_13,
            l_self_modules_backbone_modules_pag_1_modules_f_p_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_pag_1_modules_f_p_modules_conv_parameters_weight_ = (
            None
        )
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_parameters_bias_ = (None)
        mul = x_67 * f_i
        x_67 = f_i = None
        sum_1 = torch.sum(mul, dim=1)
        mul = None
        unsqueeze = sum_1.unsqueeze(1)
        sum_1 = None
        sigma = torch.sigmoid(unsqueeze)
        unsqueeze = None
        x_i_1 = torch.nn.functional.interpolate(
            x_63, size=(64, 64), mode="bilinear", align_corners=False
        )
        x_63 = None
        mul_1 = sigma * x_i_1
        x_i_1 = None
        sub = 1 - sigma
        sigma = None
        mul_2 = sub * out_13
        sub = out_13 = None
        out_15 = mul_1 + mul_2
        mul_1 = mul_2 = None
        x_68 = torch.conv2d(
            x_i,
            l_self_modules_backbone_modules_diff_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_diff_1_modules_conv_parameters_weight_ = None
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_backbone_modules_diff_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_diff_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_diff_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_diff_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = (
            l_self_modules_backbone_modules_diff_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_diff_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_diff_1_modules_bn_parameters_weight_
        ) = l_self_modules_backbone_modules_diff_1_modules_bn_parameters_bias_ = None
        interpolate_2 = torch.nn.functional.interpolate(
            x_69, size=[64, 64], mode="bilinear", align_corners=False
        )
        x_69 = None
        out_14 += interpolate_2
        x_d = out_14
        out_14 = interpolate_2 = None
        x_70 = torch.conv2d(
            x_i,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_72 = torch.nn.functional.relu(x_71, inplace=True)
        x_71 = None
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_72 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_75 = torch.conv2d(
            x_i,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_i = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_74 += x_76
        out_16 = x_74
        x_74 = x_76 = None
        out_17 = torch.nn.functional.relu(out_16, inplace=True)
        out_16 = None
        x_77 = torch.conv2d(
            out_17,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_77 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_79 = torch.nn.functional.relu(x_78, inplace=True)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_81 += out_17
        out_18 = x_81
        x_81 = out_17 = None
        out_19 = torch.nn.functional.relu(out_18, inplace=True)
        out_18 = None
        x_82 = torch.conv2d(
            out_19,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_84 = torch.nn.functional.relu(x_83, inplace=True)
        x_83 = None
        x_85 = torch.conv2d(
            x_84,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_84 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_85 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_86 += out_19
        out_20 = x_86
        x_86 = out_19 = None
        x_i_2 = torch.nn.functional.relu(out_20, inplace=False)
        out_20 = None
        residual = torch.nn.functional.relu(out_15, inplace=False)
        out_15 = None
        x_87 = torch.conv2d(
            residual,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_88 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_87 = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_89 = torch.nn.functional.relu(x_88, inplace=True)
        x_88 = None
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_91 = torch.nn.functional.batch_norm(
            x_90,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_90 = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_91 += residual
        out_21 = x_91
        x_91 = residual = None
        out_22 = torch.nn.functional.relu(out_21, inplace=True)
        out_21 = None
        x_92 = torch.conv2d(
            out_22,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_94 = torch.nn.functional.relu(x_93, inplace=True)
        x_93 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_94 = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_96 += out_22
        out_23 = x_96
        x_96 = out_22 = None
        residual_1 = torch.nn.functional.relu(x_d, inplace=False)
        x_d = None
        x_97 = torch.conv2d(
            residual_1,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_97 = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_99 = torch.nn.functional.relu(x_98, inplace=True)
        x_98 = None
        x_100 = torch.conv2d(
            x_99,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_99 = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_100 = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_102 = torch.nn.functional.relu(x_101, inplace=True)
        x_101 = None
        x_103 = torch.conv2d(
            x_102,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_102 = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_104 = torch.nn.functional.batch_norm(
            x_103,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_103 = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_105 = torch.conv2d(
            residual_1,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        residual_1 = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_106 = torch.nn.functional.batch_norm(
            x_105,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_105 = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_104 += x_106
        out_24 = x_104
        x_104 = x_106 = None
        x_107 = torch.conv2d(
            x_i_2,
            l_self_modules_backbone_modules_compression_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_compression_2_modules_conv_parameters_weight_ = (
            None
        )
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_compression_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_compression_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = l_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_compression_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_compression_2_modules_bn_parameters_bias_
        ) = None
        x_109 = torch.conv2d(
            x_108,
            l_self_modules_backbone_modules_pag_2_modules_f_i_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_pag_2_modules_f_i_modules_conv_parameters_weight_ = (
            None
        )
        x_110 = torch.nn.functional.batch_norm(
            x_109,
            l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_109 = l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_parameters_bias_ = (None)
        f_i_1 = torch.nn.functional.interpolate(
            x_110, size=(64, 64), mode="bilinear", align_corners=False
        )
        x_110 = None
        x_111 = torch.conv2d(
            out_23,
            l_self_modules_backbone_modules_pag_2_modules_f_p_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_pag_2_modules_f_p_modules_conv_parameters_weight_ = (
            None
        )
        x_112 = torch.nn.functional.batch_norm(
            x_111,
            l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_111 = l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_parameters_bias_ = (None)
        mul_3 = x_112 * f_i_1
        x_112 = f_i_1 = None
        sum_2 = torch.sum(mul_3, dim=1)
        mul_3 = None
        unsqueeze_1 = sum_2.unsqueeze(1)
        sum_2 = None
        sigma_1 = torch.sigmoid(unsqueeze_1)
        unsqueeze_1 = None
        x_i_3 = torch.nn.functional.interpolate(
            x_108, size=(64, 64), mode="bilinear", align_corners=False
        )
        x_108 = None
        mul_4 = sigma_1 * x_i_3
        x_i_3 = None
        sub_1 = 1 - sigma_1
        sigma_1 = None
        mul_5 = sub_1 * out_23
        sub_1 = out_23 = None
        out_25 = mul_4 + mul_5
        mul_4 = mul_5 = None
        x_113 = torch.conv2d(
            x_i_2,
            l_self_modules_backbone_modules_diff_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_diff_2_modules_conv_parameters_weight_ = None
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_backbone_modules_diff_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_diff_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_diff_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_diff_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = (
            l_self_modules_backbone_modules_diff_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_diff_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_diff_2_modules_bn_parameters_weight_
        ) = l_self_modules_backbone_modules_diff_2_modules_bn_parameters_bias_ = None
        interpolate_5 = torch.nn.functional.interpolate(
            x_114, size=[64, 64], mode="bilinear", align_corners=False
        )
        x_114 = None
        out_24 += interpolate_5
        x_d_1 = out_24
        out_24 = interpolate_5 = None
        x_115 = torch.conv2d(
            x_i_2,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_115 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_117 = torch.nn.functional.relu(x_116, inplace=True)
        x_116 = None
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_119 = torch.nn.functional.batch_norm(
            x_118,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_118 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_120 = torch.nn.functional.relu(x_119, inplace=True)
        x_119 = None
        x_121 = torch.conv2d(
            x_120,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_120 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_121 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_123 = torch.conv2d(
            x_i_2,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_i_2 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_122 += x_124
        out_26 = x_122
        x_122 = x_124 = None
        x_125 = torch.conv2d(
            out_26,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_125 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_127 = torch.nn.functional.relu(x_126, inplace=True)
        x_126 = None
        x_128 = torch.conv2d(
            x_127,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_127 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_130 = torch.nn.functional.relu(x_129, inplace=True)
        x_129 = None
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_130 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_132 += out_26
        out_27 = x_132
        x_132 = out_26 = None
        residual_2 = torch.nn.functional.relu(out_25, inplace=False)
        out_25 = None
        x_133 = torch.conv2d(
            residual_2,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_134 = torch.nn.functional.batch_norm(
            x_133,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_133 = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_135 = torch.nn.functional.relu(x_134, inplace=True)
        x_134 = None
        x_136 = torch.conv2d(
            x_135,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_135 = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_138 = torch.nn.functional.relu(x_137, inplace=True)
        x_137 = None
        x_139 = torch.conv2d(
            x_138,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_138 = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_139 = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_141 = torch.conv2d(
            residual_2,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        residual_2 = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_141 = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_140 += x_142
        out_28 = x_140
        x_140 = x_142 = None
        residual_3 = torch.nn.functional.relu(x_d_1, inplace=False)
        x_d_1 = None
        x_143 = torch.conv2d(
            residual_3,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_144 = torch.nn.functional.batch_norm(
            x_143,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_143 = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_145 = torch.nn.functional.relu(x_144, inplace=True)
        x_144 = None
        x_146 = torch.conv2d(
            x_145,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_145 = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_147 = torch.nn.functional.batch_norm(
            x_146,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_146 = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_148 = torch.nn.functional.relu(x_147, inplace=True)
        x_147 = None
        x_149 = torch.conv2d(
            x_148,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_148 = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_150 = torch.nn.functional.batch_norm(
            x_149,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_149 = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_151 = torch.conv2d(
            residual_3,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        residual_3 = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_151 = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_150 += x_152
        out_29 = x_150
        x_150 = x_152 = None
        x_153 = torch.nn.functional.batch_norm(
            out_27,
            l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_parameters_bias_ = (None)
        x_154 = torch.nn.functional.relu(x_153, inplace=True)
        x_153 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_154 = l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_conv_parameters_weight_ = (None)
        input_3 = torch._C._nn.avg_pool2d(out_27, 5, 2, 2, False, True, None)
        x_156 = torch.nn.functional.batch_norm(
            input_3,
            l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_3 = l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_bn_parameters_bias_ = (None)
        x_157 = torch.nn.functional.relu(x_156, inplace=True)
        x_156 = None
        x_158 = torch.conv2d(
            x_157,
            l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_157 = l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        feat_up = torch.nn.functional.interpolate(
            x_158, size=(8, 8), mode="bilinear", align_corners=False
        )
        x_158 = None
        add_2 = feat_up + x_155
        feat_up = None
        input_4 = torch._C._nn.avg_pool2d(out_27, 9, 4, 4, False, True, None)
        x_159 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_4 = l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_bn_parameters_bias_ = (None)
        x_160 = torch.nn.functional.relu(x_159, inplace=True)
        x_159 = None
        x_161 = torch.conv2d(
            x_160,
            l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_160 = l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        feat_up_1 = torch.nn.functional.interpolate(
            x_161, size=(8, 8), mode="bilinear", align_corners=False
        )
        x_161 = None
        add_3 = feat_up_1 + x_155
        feat_up_1 = None
        input_5 = torch._C._nn.avg_pool2d(out_27, 17, 8, 8, False, True, None)
        x_162 = torch.nn.functional.batch_norm(
            input_5,
            l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_5 = l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_bn_parameters_bias_ = (None)
        x_163 = torch.nn.functional.relu(x_162, inplace=True)
        x_162 = None
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_163 = l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_conv_parameters_weight_ = (None)
        feat_up_2 = torch.nn.functional.interpolate(
            x_164, size=(8, 8), mode="bilinear", align_corners=False
        )
        x_164 = None
        add_4 = feat_up_2 + x_155
        feat_up_2 = None
        input_6 = torch.nn.functional.adaptive_avg_pool2d(out_27, (1, 1))
        x_165 = torch.nn.functional.batch_norm(
            input_6,
            l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_6 = l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_bn_parameters_bias_ = (None)
        x_166 = torch.nn.functional.relu(x_165, inplace=True)
        x_165 = None
        x_167 = torch.conv2d(
            x_166,
            l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_166 = l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_conv_parameters_weight_ = (None)
        feat_up_3 = torch.nn.functional.interpolate(
            x_167, size=(8, 8), mode="bilinear", align_corners=False
        )
        x_167 = None
        add_5 = feat_up_3 + x_155
        feat_up_3 = None
        cat = torch.cat([add_2, add_3, add_4, add_5], dim=1)
        add_2 = add_3 = add_4 = add_5 = None
        x_168 = torch.nn.functional.batch_norm(
            cat,
            l_self_modules_backbone_modules_spp_modules_processes_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        cat = l_self_modules_backbone_modules_spp_modules_processes_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_processes_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_processes_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_processes_modules_bn_parameters_bias_ = (None)
        x_169 = torch.nn.functional.relu(x_168, inplace=True)
        x_168 = None
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_backbone_modules_spp_modules_processes_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        x_169 = l_self_modules_backbone_modules_spp_modules_processes_modules_conv_parameters_weight_ = (None)
        cat_1 = torch.cat([x_155, x_170], dim=1)
        x_155 = x_170 = None
        x_171 = torch.nn.functional.batch_norm(
            cat_1,
            l_self_modules_backbone_modules_spp_modules_compression_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_compression_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_compression_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_compression_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        cat_1 = l_self_modules_backbone_modules_spp_modules_compression_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_compression_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_compression_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_compression_modules_bn_parameters_bias_ = (None)
        x_172 = torch.nn.functional.relu(x_171, inplace=True)
        x_171 = None
        x_173 = torch.conv2d(
            x_172,
            l_self_modules_backbone_modules_spp_modules_compression_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_172 = l_self_modules_backbone_modules_spp_modules_compression_modules_conv_parameters_weight_ = (None)
        x_174 = torch.nn.functional.batch_norm(
            out_27,
            l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_27 = l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_parameters_bias_ = (None)
        x_175 = torch.nn.functional.relu(x_174, inplace=True)
        x_174 = None
        x_176 = torch.conv2d(
            x_175,
            l_self_modules_backbone_modules_spp_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_175 = l_self_modules_backbone_modules_spp_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_i_4 = x_173 + x_176
        x_173 = x_176 = None
        x_i_5 = torch.nn.functional.interpolate(
            x_i_4, size=[64, 64], mode="bilinear", align_corners=False
        )
        x_i_4 = None
        sigma_2 = torch.sigmoid(out_29)
        out_29 = None
        sub_2 = 1 - sigma_2
        mul_6 = sub_2 * x_i_5
        sub_2 = None
        add_7 = mul_6 + out_28
        mul_6 = None
        x_177 = torch.conv2d(
            add_7,
            l_self_modules_backbone_modules_dfm_modules_f_p_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        add_7 = l_self_modules_backbone_modules_dfm_modules_f_p_modules_conv_parameters_weight_ = (None)
        x_178 = torch.nn.functional.batch_norm(
            x_177,
            l_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_177 = l_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_dfm_modules_f_p_modules_bn_parameters_bias_
        ) = None
        mul_7 = sigma_2 * out_28
        sigma_2 = out_28 = None
        add_8 = x_i_5 + mul_7
        x_i_5 = mul_7 = None
        x_179 = torch.conv2d(
            add_8,
            l_self_modules_backbone_modules_dfm_modules_f_i_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        add_8 = l_self_modules_backbone_modules_dfm_modules_f_i_modules_conv_parameters_weight_ = (None)
        x_180 = torch.nn.functional.batch_norm(
            x_179,
            l_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_179 = l_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_dfm_modules_f_i_modules_bn_parameters_bias_
        ) = None
        out_30 = x_178 + x_180
        x_178 = x_180 = None
        x_181 = torch.nn.functional.batch_norm(
            out_30,
            l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_30 = l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_parameters_bias_ = (None)
        x_182 = torch.nn.functional.relu(x_181, inplace=True)
        x_181 = None
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_decode_head_modules_i_head_modules_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_182 = l_self_modules_decode_head_modules_i_head_modules_conv_modules_conv_parameters_weight_ = (None)
        x_184 = torch.nn.functional.batch_norm(
            x_183,
            l_self_modules_decode_head_modules_i_head_modules_norm_buffers_running_mean_,
            l_self_modules_decode_head_modules_i_head_modules_norm_buffers_running_var_,
            l_self_modules_decode_head_modules_i_head_modules_norm_parameters_weight_,
            l_self_modules_decode_head_modules_i_head_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_183 = (
            l_self_modules_decode_head_modules_i_head_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_decode_head_modules_i_head_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_decode_head_modules_i_head_modules_norm_parameters_weight_
        ) = (
            l_self_modules_decode_head_modules_i_head_modules_norm_parameters_bias_
        ) = None
        x_185 = torch.nn.functional.relu(x_184, inplace=True)
        x_184 = None
        feat = torch.nn.functional.dropout2d(x_185, 0.1, False, False)
        x_185 = None
        output = torch.conv2d(
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
        return (output,)
