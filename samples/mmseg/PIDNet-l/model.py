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
        L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_backbone_modules_dfm_modules_conv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_dfm_modules_conv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_dfm_modules_conv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_dfm_modules_conv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_dfm_modules_conv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_parameters_bias_
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
        l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_parameters_bias_
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
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_parameters_bias_
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
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_bias_
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
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_parameters_bias_
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
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_parameters_bias_
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
        l_self_modules_backbone_modules_dfm_modules_conv_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_dfm_modules_conv_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_dfm_modules_conv_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_dfm_modules_conv_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_dfm_modules_conv_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_dfm_modules_conv_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_dfm_modules_conv_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_dfm_modules_conv_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_dfm_modules_conv_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_dfm_modules_conv_modules_conv_parameters_weight_
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
        out_3 = torch.nn.functional.relu(out_2, inplace=True)
        out_2 = None
        x_16 = torch.conv2d(
            out_3,
            l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_18 = torch.nn.functional.relu(x_17, inplace=True)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_2_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_20 += out_3
        out_4 = x_20
        x_20 = out_3 = None
        input_1 = torch.nn.functional.relu(out_4, inplace=False)
        out_4 = None
        x_21 = torch.conv2d(
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
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_21 = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_23 = torch.nn.functional.relu(x_22, inplace=True)
        x_22 = None
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_23 = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_24 = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_26 = torch.conv2d(
            input_1,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        input_1 = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_4_modules_0_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_25 += x_27
        out_5 = x_25
        x_25 = x_27 = None
        out_6 = torch.nn.functional.relu(out_5, inplace=True)
        out_5 = None
        x_28 = torch.conv2d(
            out_6,
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
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_30 = torch.nn.functional.relu(x_29, inplace=True)
        x_29 = None
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_30 = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_4_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_32 += out_6
        out_7 = x_32
        x_32 = out_6 = None
        out_8 = torch.nn.functional.relu(out_7, inplace=True)
        out_7 = None
        x_33 = torch.conv2d(
            out_8,
            l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_35 = torch.nn.functional.relu(x_34, inplace=True)
        x_34 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_stem_modules_4_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_37 += out_8
        out_9 = x_37
        x_37 = out_8 = None
        input_2 = torch.nn.functional.relu(out_9, inplace=False)
        out_9 = None
        x_38 = torch.conv2d(
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
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_40 = torch.nn.functional.relu(x_39, inplace=True)
        x_39 = None
        x_41 = torch.conv2d(
            x_40,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_40 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_42 = torch.nn.functional.batch_norm(
            x_41,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_41 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_43 = torch.conv2d(
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
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_0_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_42 += x_44
        out_10 = x_42
        x_42 = x_44 = None
        out_11 = torch.nn.functional.relu(out_10, inplace=True)
        out_10 = None
        x_45 = torch.conv2d(
            out_11,
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
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_47 = torch.nn.functional.relu(x_46, inplace=True)
        x_46 = None
        x_48 = torch.conv2d(
            x_47,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_47 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_48 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_49 += out_11
        out_12 = x_49
        x_49 = out_11 = None
        out_13 = torch.nn.functional.relu(out_12, inplace=True)
        out_12 = None
        x_50 = torch.conv2d(
            out_13,
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
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_54 += out_13
        out_14 = x_54
        x_54 = out_13 = None
        out_15 = torch.nn.functional.relu(out_14, inplace=True)
        out_14 = None
        x_55 = torch.conv2d(
            out_15,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_55 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_57 = torch.nn.functional.relu(x_56, inplace=True)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_0_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_59 += out_15
        out_16 = x_59
        x_59 = out_15 = None
        x_i = torch.nn.functional.relu(out_16, inplace=False)
        out_16 = None
        x_60 = torch.conv2d(
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
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_60 = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_62 = torch.nn.functional.relu(x_61, inplace=True)
        x_61 = None
        x_63 = torch.conv2d(
            x_62,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_62 = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_64 += input_2
        out_17 = x_64
        x_64 = None
        out_18 = torch.nn.functional.relu(out_17, inplace=True)
        out_17 = None
        x_65 = torch.conv2d(
            out_18,
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
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_65 = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_67 = torch.nn.functional.relu(x_66, inplace=True)
        x_66 = None
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_69 += out_18
        out_19 = x_69
        x_69 = out_18 = None
        out_20 = torch.nn.functional.relu(out_19, inplace=True)
        out_19 = None
        x_70 = torch.conv2d(
            out_20,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_72 = torch.nn.functional.relu(x_71, inplace=True)
        x_71 = None
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_72 = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_0_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_74 += out_20
        out_21 = x_74
        x_74 = out_20 = None
        x_75 = torch.conv2d(
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
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_77 = torch.nn.functional.relu(x_76, inplace=True)
        x_76 = None
        x_78 = torch.conv2d(
            x_77,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_77 = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_79 = torch.nn.functional.batch_norm(
            x_78,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_78 = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_79 += input_2
        out_22 = x_79
        x_79 = input_2 = None
        x_80 = torch.conv2d(
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
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_compression_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_compression_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = l_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_compression_1_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_compression_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_compression_1_modules_bn_parameters_bias_
        ) = None
        x_82 = torch.conv2d(
            x_81,
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
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_pag_1_modules_f_i_modules_bn_parameters_bias_ = (None)
        f_i = torch.nn.functional.interpolate(
            x_83, size=(64, 64), mode="bilinear", align_corners=False
        )
        x_83 = None
        x_84 = torch.conv2d(
            out_21,
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
        x_85 = torch.nn.functional.batch_norm(
            x_84,
            l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_84 = l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_pag_1_modules_f_p_modules_bn_parameters_bias_ = (None)
        mul = x_85 * f_i
        x_85 = f_i = None
        sum_1 = torch.sum(mul, dim=1)
        mul = None
        unsqueeze = sum_1.unsqueeze(1)
        sum_1 = None
        sigma = torch.sigmoid(unsqueeze)
        unsqueeze = None
        x_i_1 = torch.nn.functional.interpolate(
            x_81, size=(64, 64), mode="bilinear", align_corners=False
        )
        x_81 = None
        mul_1 = sigma * x_i_1
        x_i_1 = None
        sub = 1 - sigma
        sigma = None
        mul_2 = sub * out_21
        sub = out_21 = None
        out_23 = mul_1 + mul_2
        mul_1 = mul_2 = None
        x_86 = torch.conv2d(
            x_i,
            l_self_modules_backbone_modules_diff_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_diff_1_modules_conv_parameters_weight_ = None
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_backbone_modules_diff_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_diff_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_diff_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_diff_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = (
            l_self_modules_backbone_modules_diff_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_diff_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_diff_1_modules_bn_parameters_weight_
        ) = l_self_modules_backbone_modules_diff_1_modules_bn_parameters_bias_ = None
        interpolate_2 = torch.nn.functional.interpolate(
            x_87, size=[64, 64], mode="bilinear", align_corners=False
        )
        x_87 = None
        out_22 += interpolate_2
        x_d = out_22
        out_22 = interpolate_2 = None
        x_88 = torch.conv2d(
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
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_90 = torch.nn.functional.relu(x_89, inplace=True)
        x_89 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_90 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_93 = torch.conv2d(
            x_i,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_i = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_94 = torch.nn.functional.batch_norm(
            x_93,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_93 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_0_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_92 += x_94
        out_24 = x_92
        x_92 = x_94 = None
        out_25 = torch.nn.functional.relu(out_24, inplace=True)
        out_24 = None
        x_95 = torch.conv2d(
            out_25,
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
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_97 = torch.nn.functional.relu(x_96, inplace=True)
        x_96 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_98 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_99 += out_25
        out_26 = x_99
        x_99 = out_25 = None
        out_27 = torch.nn.functional.relu(out_26, inplace=True)
        out_26 = None
        x_100 = torch.conv2d(
            out_27,
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
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_100 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_102 = torch.nn.functional.relu(x_101, inplace=True)
        x_101 = None
        x_103 = torch.conv2d(
            x_102,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_102 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_104 = torch.nn.functional.batch_norm(
            x_103,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_103 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_104 += out_27
        out_28 = x_104
        x_104 = out_27 = None
        out_29 = torch.nn.functional.relu(out_28, inplace=True)
        out_28 = None
        x_105 = torch.conv2d(
            out_29,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_106 = torch.nn.functional.batch_norm(
            x_105,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_105 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_107 = torch.nn.functional.relu(x_106, inplace=True)
        x_106 = None
        x_108 = torch.conv2d(
            x_107,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_107 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_108 = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_1_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_109 += out_29
        out_30 = x_109
        x_109 = out_29 = None
        x_i_2 = torch.nn.functional.relu(out_30, inplace=False)
        out_30 = None
        residual = torch.nn.functional.relu(out_23, inplace=False)
        out_23 = None
        x_110 = torch.conv2d(
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
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_112 = torch.nn.functional.relu(x_111, inplace=True)
        x_111 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_112 = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_114 += residual
        out_31 = x_114
        x_114 = residual = None
        out_32 = torch.nn.functional.relu(out_31, inplace=True)
        out_31 = None
        x_115 = torch.conv2d(
            out_32,
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
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_115 = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_117 = torch.nn.functional.relu(x_116, inplace=True)
        x_116 = None
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_119 = torch.nn.functional.batch_norm(
            x_118,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_118 = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_119 += out_32
        out_33 = x_119
        x_119 = out_32 = None
        out_34 = torch.nn.functional.relu(out_33, inplace=True)
        out_33 = None
        x_120 = torch.conv2d(
            out_34,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_122 = torch.nn.functional.relu(x_121, inplace=True)
        x_121 = None
        x_123 = torch.conv2d(
            x_122,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_122 = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_1_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_124 += out_34
        out_35 = x_124
        x_124 = out_34 = None
        residual_1 = torch.nn.functional.relu(x_d, inplace=False)
        x_d = None
        x_125 = torch.conv2d(
            residual_1,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_125 = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_127 = torch.nn.functional.relu(x_126, inplace=True)
        x_126 = None
        x_128 = torch.conv2d(
            x_127,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_127 = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_129 += residual_1
        out_36 = x_129
        x_129 = residual_1 = None
        x_130 = torch.conv2d(
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
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_compression_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_compression_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_130 = l_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_compression_2_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_compression_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_compression_2_modules_bn_parameters_bias_
        ) = None
        x_132 = torch.conv2d(
            x_131,
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
        x_133 = torch.nn.functional.batch_norm(
            x_132,
            l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_132 = l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_pag_2_modules_f_i_modules_bn_parameters_bias_ = (None)
        f_i_1 = torch.nn.functional.interpolate(
            x_133, size=(64, 64), mode="bilinear", align_corners=False
        )
        x_133 = None
        x_134 = torch.conv2d(
            out_35,
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
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_pag_2_modules_f_p_modules_bn_parameters_bias_ = (None)
        mul_3 = x_135 * f_i_1
        x_135 = f_i_1 = None
        sum_2 = torch.sum(mul_3, dim=1)
        mul_3 = None
        unsqueeze_1 = sum_2.unsqueeze(1)
        sum_2 = None
        sigma_1 = torch.sigmoid(unsqueeze_1)
        unsqueeze_1 = None
        x_i_3 = torch.nn.functional.interpolate(
            x_131, size=(64, 64), mode="bilinear", align_corners=False
        )
        x_131 = None
        mul_4 = sigma_1 * x_i_3
        x_i_3 = None
        sub_1 = 1 - sigma_1
        sigma_1 = None
        mul_5 = sub_1 * out_35
        sub_1 = out_35 = None
        out_37 = mul_4 + mul_5
        mul_4 = mul_5 = None
        x_136 = torch.conv2d(
            x_i_2,
            l_self_modules_backbone_modules_diff_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_diff_2_modules_conv_parameters_weight_ = None
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_backbone_modules_diff_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_diff_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_diff_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_diff_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = (
            l_self_modules_backbone_modules_diff_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_diff_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_diff_2_modules_bn_parameters_weight_
        ) = l_self_modules_backbone_modules_diff_2_modules_bn_parameters_bias_ = None
        interpolate_5 = torch.nn.functional.interpolate(
            x_137, size=[64, 64], mode="bilinear", align_corners=False
        )
        x_137 = None
        out_36 += interpolate_5
        x_d_1 = out_36
        out_36 = interpolate_5 = None
        x_138 = torch.conv2d(
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
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_138 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_140 = torch.nn.functional.relu(x_139, inplace=True)
        x_139 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_140 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_141 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_143 = torch.nn.functional.relu(x_142, inplace=True)
        x_142 = None
        x_144 = torch.conv2d(
            x_143,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_143 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_145 = torch.nn.functional.batch_norm(
            x_144,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_144 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_146 = torch.conv2d(
            x_i_2,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_i_2 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_147 = torch.nn.functional.batch_norm(
            x_146,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_146 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_145 += x_147
        out_38 = x_145
        x_145 = x_147 = None
        x_148 = torch.conv2d(
            out_38,
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
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_150 = torch.nn.functional.relu(x_149, inplace=True)
        x_149 = None
        x_151 = torch.conv2d(
            x_150,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_150 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_151 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_153 = torch.nn.functional.relu(x_152, inplace=True)
        x_152 = None
        x_154 = torch.conv2d(
            x_153,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_153 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_i_branch_layers_modules_2_modules_1_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_155 += out_38
        out_39 = x_155
        x_155 = out_38 = None
        residual_2 = torch.nn.functional.relu(out_37, inplace=False)
        out_37 = None
        x_156 = torch.conv2d(
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
        x_157 = torch.nn.functional.batch_norm(
            x_156,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_156 = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_158 = torch.nn.functional.relu(x_157, inplace=True)
        x_157 = None
        x_159 = torch.conv2d(
            x_158,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_158 = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_160 = torch.nn.functional.batch_norm(
            x_159,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_159 = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_161 = torch.nn.functional.relu(x_160, inplace=True)
        x_160 = None
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_161 = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_163 = torch.nn.functional.batch_norm(
            x_162,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_162 = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_164 = torch.conv2d(
            residual_2,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        residual_2 = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_165 = torch.nn.functional.batch_norm(
            x_164,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_164 = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_p_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_163 += x_165
        out_40 = x_163
        x_163 = x_165 = None
        residual_3 = torch.nn.functional.relu(x_d_1, inplace=False)
        x_d_1 = None
        x_166 = torch.conv2d(
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
        x_167 = torch.nn.functional.batch_norm(
            x_166,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_166 = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_168 = torch.nn.functional.relu(x_167, inplace=True)
        x_167 = None
        x_169 = torch.conv2d(
            x_168,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_168 = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_170 = torch.nn.functional.batch_norm(
            x_169,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_169 = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_171 = torch.nn.functional.relu(x_170, inplace=True)
        x_170 = None
        x_172 = torch.conv2d(
            x_171,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_171 = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_173 = torch.nn.functional.batch_norm(
            x_172,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_172 = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_174 = torch.conv2d(
            residual_3,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        residual_3 = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_175 = torch.nn.functional.batch_norm(
            x_174,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_174 = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_d_branch_layers_modules_2_modules_0_modules_downsample_modules_bn_parameters_bias_ = (None)
        x_173 += x_175
        out_41 = x_173
        x_173 = x_175 = None
        x_176 = torch.nn.functional.batch_norm(
            out_39,
            l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_bn_parameters_bias_ = (None)
        x_177 = torch.nn.functional.relu(x_176, inplace=True)
        x_176 = None
        x_178 = torch.conv2d(
            x_177,
            l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_177 = l_self_modules_backbone_modules_spp_modules_scales_modules_0_modules_conv_parameters_weight_ = (None)
        input_3 = torch._C._nn.avg_pool2d(out_39, 5, 2, 2, False, True, None)
        x_179 = torch.nn.functional.batch_norm(
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
        x_180 = torch.nn.functional.relu(x_179, inplace=True)
        x_179 = None
        x_181 = torch.conv2d(
            x_180,
            l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_180 = l_self_modules_backbone_modules_spp_modules_scales_modules_1_modules_1_modules_conv_parameters_weight_ = (None)
        feat_up = torch.nn.functional.interpolate(x_181, size=(8, 8), mode="bilinear")
        x_181 = None
        add_2 = feat_up + x_178
        feat_up = None
        x_182 = torch.nn.functional.batch_norm(
            add_2,
            l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        add_2 = l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_bn_parameters_bias_ = (None)
        x_183 = torch.nn.functional.relu(x_182, inplace=True)
        x_182 = None
        x_184 = torch.conv2d(
            x_183,
            l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_183 = l_self_modules_backbone_modules_spp_modules_processes_modules_0_modules_conv_parameters_weight_ = (None)
        input_4 = torch._C._nn.avg_pool2d(out_39, 9, 4, 4, False, True, None)
        x_185 = torch.nn.functional.batch_norm(
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
        x_186 = torch.nn.functional.relu(x_185, inplace=True)
        x_185 = None
        x_187 = torch.conv2d(
            x_186,
            l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_186 = l_self_modules_backbone_modules_spp_modules_scales_modules_2_modules_1_modules_conv_parameters_weight_ = (None)
        feat_up_1 = torch.nn.functional.interpolate(x_187, size=(8, 8), mode="bilinear")
        x_187 = None
        add_3 = feat_up_1 + x_184
        feat_up_1 = None
        x_188 = torch.nn.functional.batch_norm(
            add_3,
            l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        add_3 = l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_bn_parameters_bias_ = (None)
        x_189 = torch.nn.functional.relu(x_188, inplace=True)
        x_188 = None
        x_190 = torch.conv2d(
            x_189,
            l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_189 = l_self_modules_backbone_modules_spp_modules_processes_modules_1_modules_conv_parameters_weight_ = (None)
        input_5 = torch._C._nn.avg_pool2d(out_39, 17, 8, 8, False, True, None)
        x_191 = torch.nn.functional.batch_norm(
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
        x_192 = torch.nn.functional.relu(x_191, inplace=True)
        x_191 = None
        x_193 = torch.conv2d(
            x_192,
            l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_192 = l_self_modules_backbone_modules_spp_modules_scales_modules_3_modules_1_modules_conv_parameters_weight_ = (None)
        feat_up_2 = torch.nn.functional.interpolate(x_193, size=(8, 8), mode="bilinear")
        x_193 = None
        add_4 = feat_up_2 + x_190
        feat_up_2 = None
        x_194 = torch.nn.functional.batch_norm(
            add_4,
            l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        add_4 = l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_bn_parameters_bias_ = (None)
        x_195 = torch.nn.functional.relu(x_194, inplace=True)
        x_194 = None
        x_196 = torch.conv2d(
            x_195,
            l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_195 = l_self_modules_backbone_modules_spp_modules_processes_modules_2_modules_conv_parameters_weight_ = (None)
        input_6 = torch.nn.functional.adaptive_avg_pool2d(out_39, (1, 1))
        x_197 = torch.nn.functional.batch_norm(
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
        x_198 = torch.nn.functional.relu(x_197, inplace=True)
        x_197 = None
        x_199 = torch.conv2d(
            x_198,
            l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_198 = l_self_modules_backbone_modules_spp_modules_scales_modules_4_modules_1_modules_conv_parameters_weight_ = (None)
        feat_up_3 = torch.nn.functional.interpolate(x_199, size=(8, 8), mode="bilinear")
        x_199 = None
        add_5 = feat_up_3 + x_196
        feat_up_3 = None
        x_200 = torch.nn.functional.batch_norm(
            add_5,
            l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        add_5 = l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_bn_parameters_bias_ = (None)
        x_201 = torch.nn.functional.relu(x_200, inplace=True)
        x_200 = None
        x_202 = torch.conv2d(
            x_201,
            l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_201 = l_self_modules_backbone_modules_spp_modules_processes_modules_3_modules_conv_parameters_weight_ = (None)
        cat = torch.cat([x_178, x_184, x_190, x_196, x_202], dim=1)
        x_178 = x_184 = x_190 = x_196 = x_202 = None
        x_203 = torch.nn.functional.batch_norm(
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
        x_204 = torch.nn.functional.relu(x_203, inplace=True)
        x_203 = None
        x_205 = torch.conv2d(
            x_204,
            l_self_modules_backbone_modules_spp_modules_compression_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_204 = l_self_modules_backbone_modules_spp_modules_compression_modules_conv_parameters_weight_ = (None)
        x_206 = torch.nn.functional.batch_norm(
            out_39,
            l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_39 = l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_spp_modules_shortcut_modules_bn_parameters_bias_ = (None)
        x_207 = torch.nn.functional.relu(x_206, inplace=True)
        x_206 = None
        x_208 = torch.conv2d(
            x_207,
            l_self_modules_backbone_modules_spp_modules_shortcut_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_207 = l_self_modules_backbone_modules_spp_modules_shortcut_modules_conv_parameters_weight_ = (None)
        x_i_4 = x_205 + x_208
        x_205 = x_208 = None
        x_i_5 = torch.nn.functional.interpolate(
            x_i_4, size=[64, 64], mode="bilinear", align_corners=False
        )
        x_i_4 = None
        sigma_2 = torch.sigmoid(out_41)
        out_41 = None
        mul_6 = sigma_2 * out_40
        out_40 = None
        sub_2 = 1 - sigma_2
        sigma_2 = None
        mul_7 = sub_2 * x_i_5
        sub_2 = x_i_5 = None
        add_7 = mul_6 + mul_7
        mul_6 = mul_7 = None
        x_209 = torch.nn.functional.batch_norm(
            add_7,
            l_self_modules_backbone_modules_dfm_modules_conv_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_dfm_modules_conv_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_dfm_modules_conv_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_dfm_modules_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        add_7 = l_self_modules_backbone_modules_dfm_modules_conv_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_dfm_modules_conv_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_dfm_modules_conv_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_dfm_modules_conv_modules_bn_parameters_bias_
        ) = None
        x_210 = torch.nn.functional.relu(x_209, inplace=True)
        x_209 = None
        x_211 = torch.conv2d(
            x_210,
            l_self_modules_backbone_modules_dfm_modules_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_210 = l_self_modules_backbone_modules_dfm_modules_conv_modules_conv_parameters_weight_ = (None)
        x_212 = torch.nn.functional.batch_norm(
            x_211,
            l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_211 = l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_i_head_modules_conv_modules_bn_parameters_bias_ = (None)
        x_213 = torch.nn.functional.relu(x_212, inplace=True)
        x_212 = None
        x_214 = torch.conv2d(
            x_213,
            l_self_modules_decode_head_modules_i_head_modules_conv_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_213 = l_self_modules_decode_head_modules_i_head_modules_conv_modules_conv_parameters_weight_ = (None)
        x_215 = torch.nn.functional.batch_norm(
            x_214,
            l_self_modules_decode_head_modules_i_head_modules_norm_buffers_running_mean_,
            l_self_modules_decode_head_modules_i_head_modules_norm_buffers_running_var_,
            l_self_modules_decode_head_modules_i_head_modules_norm_parameters_weight_,
            l_self_modules_decode_head_modules_i_head_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_214 = (
            l_self_modules_decode_head_modules_i_head_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_decode_head_modules_i_head_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_decode_head_modules_i_head_modules_norm_parameters_weight_
        ) = (
            l_self_modules_decode_head_modules_i_head_modules_norm_parameters_bias_
        ) = None
        x_216 = torch.nn.functional.relu(x_215, inplace=True)
        x_215 = None
        feat = torch.nn.functional.dropout2d(x_216, 0.1, False, False)
        x_216 = None
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
