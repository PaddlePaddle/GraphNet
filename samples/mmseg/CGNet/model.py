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
        L_self_modules_backbone_modules_stem_modules_0_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_stem_modules_2_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm_prelu_0_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_norm_prelu_0_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_norm_prelu_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm_prelu_0_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm_prelu_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_0_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_0_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_0_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_0_modules_bottleneck_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_1_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_1_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_2_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_2_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_2_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm_prelu_1_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_norm_prelu_1_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_norm_prelu_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm_prelu_1_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm_prelu_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_0_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_0_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_0_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_0_modules_bottleneck_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_1_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_1_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_2_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_2_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_2_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_3_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_3_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_3_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_4_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_4_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_4_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_5_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_5_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_5_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_6_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_6_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_6_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_6_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_6_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_6_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_6_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_7_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_7_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_7_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_7_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_7_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_7_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_7_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_8_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_8_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_8_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_8_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_8_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_8_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_8_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_9_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_9_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_9_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_9_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_9_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_9_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_9_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_10_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_10_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_10_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_10_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_10_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_10_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_10_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_11_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_11_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_11_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_11_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_11_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_11_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_11_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_12_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_12_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_12_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_12_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_12_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_12_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_12_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_13_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_13_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_13_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_13_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_13_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_13_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_13_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_14_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_14_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_14_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_14_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_14_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_14_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_14_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_15_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_15_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_15_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_15_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_15_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_15_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_15_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_16_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_16_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_16_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_16_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_16_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_16_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_16_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_17_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_17_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_17_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_17_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_17_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_17_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_17_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_18_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_18_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_18_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_18_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_18_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_18_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_18_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_19_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_19_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_19_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_19_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_19_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_19_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_19_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_20_modules_f_loc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_20_modules_f_sur_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_20_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_20_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_level2_modules_20_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_20_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_20_modules_activate_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm_prelu_2_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_norm_prelu_2_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_norm_prelu_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm_prelu_2_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_norm_prelu_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_backbone_modules_stem_modules_0_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_0_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_stem_modules_1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_2_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_2_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_weight_ = (
            L_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_weight_
        )
        l_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_stem_modules_2_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_stem_modules_2_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_norm_prelu_0_modules_0_buffers_running_mean_ = (
            L_self_modules_backbone_modules_norm_prelu_0_modules_0_buffers_running_mean_
        )
        l_self_modules_backbone_modules_norm_prelu_0_modules_0_buffers_running_var_ = (
            L_self_modules_backbone_modules_norm_prelu_0_modules_0_buffers_running_var_
        )
        l_self_modules_backbone_modules_norm_prelu_0_modules_0_parameters_weight_ = (
            L_self_modules_backbone_modules_norm_prelu_0_modules_0_parameters_weight_
        )
        l_self_modules_backbone_modules_norm_prelu_0_modules_0_parameters_bias_ = (
            L_self_modules_backbone_modules_norm_prelu_0_modules_0_parameters_bias_
        )
        l_self_modules_backbone_modules_norm_prelu_0_modules_1_parameters_weight_ = (
            L_self_modules_backbone_modules_norm_prelu_0_modules_1_parameters_weight_
        )
        l_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_0_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_0_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_0_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_0_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level1_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_level1_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_level1_modules_0_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_0_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_0_modules_bottleneck_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_0_modules_bottleneck_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_1_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_1_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_1_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_1_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level1_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_level1_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_level1_modules_1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_2_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_2_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_2_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_2_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level1_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_level1_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_level1_modules_2_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_2_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_norm_prelu_1_modules_0_buffers_running_mean_ = (
            L_self_modules_backbone_modules_norm_prelu_1_modules_0_buffers_running_mean_
        )
        l_self_modules_backbone_modules_norm_prelu_1_modules_0_buffers_running_var_ = (
            L_self_modules_backbone_modules_norm_prelu_1_modules_0_buffers_running_var_
        )
        l_self_modules_backbone_modules_norm_prelu_1_modules_0_parameters_weight_ = (
            L_self_modules_backbone_modules_norm_prelu_1_modules_0_parameters_weight_
        )
        l_self_modules_backbone_modules_norm_prelu_1_modules_0_parameters_bias_ = (
            L_self_modules_backbone_modules_norm_prelu_1_modules_0_parameters_bias_
        )
        l_self_modules_backbone_modules_norm_prelu_1_modules_1_parameters_weight_ = (
            L_self_modules_backbone_modules_norm_prelu_1_modules_1_parameters_weight_
        )
        l_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_0_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_0_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_0_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_0_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_level2_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_level2_modules_0_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_0_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_0_modules_bottleneck_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_0_modules_bottleneck_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_1_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_1_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_1_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_1_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_level2_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_level2_modules_1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_2_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_2_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_2_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_2_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_2_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_2_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_level2_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_level2_modules_2_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_2_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_3_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_3_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_3_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_3_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_3_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_3_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_3_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_3_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_3_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_level2_modules_3_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_level2_modules_3_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_3_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_4_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_4_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_4_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_4_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_4_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_4_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_4_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_4_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_4_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_level2_modules_4_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_level2_modules_4_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_4_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_5_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_5_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_5_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_5_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_5_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_5_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_5_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_5_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_5_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_5_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_5_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_level2_modules_5_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_level2_modules_5_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_5_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_6_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_6_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_6_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_6_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_6_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_6_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_6_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_6_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_6_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_6_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_6_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_level2_modules_6_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_level2_modules_6_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_6_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_7_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_7_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_7_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_7_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_7_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_7_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_7_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_7_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_7_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_7_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_7_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_level2_modules_7_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_level2_modules_7_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_7_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_8_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_8_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_8_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_8_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_8_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_8_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_8_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_8_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_8_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_8_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_8_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_level2_modules_8_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_level2_modules_8_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_8_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_9_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_9_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_9_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_9_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_9_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_9_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_9_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_9_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_9_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_9_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_9_modules_bn_parameters_bias_ = (
            L_self_modules_backbone_modules_level2_modules_9_modules_bn_parameters_bias_
        )
        l_self_modules_backbone_modules_level2_modules_9_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_9_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_10_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_10_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_10_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_10_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_10_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_10_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_10_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_10_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_10_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_10_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_10_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_10_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_10_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_10_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_11_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_11_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_11_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_11_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_11_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_11_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_11_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_11_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_11_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_11_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_11_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_11_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_11_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_11_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_12_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_12_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_12_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_12_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_12_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_12_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_12_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_12_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_12_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_12_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_12_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_12_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_12_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_12_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_13_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_13_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_13_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_13_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_13_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_13_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_13_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_13_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_13_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_13_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_13_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_13_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_13_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_13_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_14_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_14_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_14_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_14_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_14_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_14_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_14_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_14_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_14_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_14_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_14_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_14_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_14_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_14_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_15_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_15_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_15_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_15_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_15_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_15_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_15_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_15_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_15_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_15_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_15_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_15_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_15_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_15_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_16_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_16_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_16_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_16_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_16_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_16_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_16_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_16_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_16_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_16_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_16_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_16_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_16_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_16_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_17_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_17_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_17_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_17_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_17_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_17_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_17_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_17_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_17_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_17_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_17_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_17_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_17_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_17_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_18_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_18_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_18_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_18_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_18_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_18_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_18_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_18_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_18_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_18_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_18_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_18_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_18_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_18_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_19_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_19_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_19_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_19_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_19_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_19_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_19_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_19_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_19_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_19_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_19_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_19_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_19_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_19_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_20_modules_f_loc_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_20_modules_f_loc_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_20_modules_f_sur_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_20_modules_f_sur_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_20_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_level2_modules_20_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_level2_modules_20_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_level2_modules_20_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_level2_modules_20_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_20_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_20_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_20_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_20_modules_activate_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_20_modules_activate_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_0_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_0_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_0_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_0_parameters_bias_
        l_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_2_parameters_weight_ = L_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_2_parameters_weight_
        l_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_2_parameters_bias_ = L_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_2_parameters_bias_
        l_self_modules_backbone_modules_norm_prelu_2_modules_0_buffers_running_mean_ = (
            L_self_modules_backbone_modules_norm_prelu_2_modules_0_buffers_running_mean_
        )
        l_self_modules_backbone_modules_norm_prelu_2_modules_0_buffers_running_var_ = (
            L_self_modules_backbone_modules_norm_prelu_2_modules_0_buffers_running_var_
        )
        l_self_modules_backbone_modules_norm_prelu_2_modules_0_parameters_weight_ = (
            L_self_modules_backbone_modules_norm_prelu_2_modules_0_parameters_weight_
        )
        l_self_modules_backbone_modules_norm_prelu_2_modules_0_parameters_bias_ = (
            L_self_modules_backbone_modules_norm_prelu_2_modules_0_parameters_bias_
        )
        l_self_modules_backbone_modules_norm_prelu_2_modules_1_parameters_weight_ = (
            L_self_modules_backbone_modules_norm_prelu_2_modules_1_parameters_weight_
        )
        l_self_modules_decode_head_modules_conv_seg_parameters_weight_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_weight_
        )
        l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_bias_
        )
        x = torch._C._nn.avg_pool2d(l_inputs_, 3, 2, 1, False, True, None)
        x_1 = torch._C._nn.avg_pool2d(l_inputs_, 3, 2, 1, False, True, None)
        x_2 = torch._C._nn.avg_pool2d(x_1, 3, 2, 1, False, True, None)
        x_1 = None
        x_3 = torch.conv2d(
            l_inputs_,
            l_self_modules_backbone_modules_stem_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_inputs_ = l_self_modules_backbone_modules_stem_modules_0_modules_conv_parameters_weight_ = (None)
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_3 = l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_0_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_stem_modules_0_modules_bn_parameters_bias_
        ) = None
        x_5 = torch.prelu(
            x_4,
            l_self_modules_backbone_modules_stem_modules_0_modules_activate_parameters_weight_,
        )
        x_4 = l_self_modules_backbone_modules_stem_modules_0_modules_activate_parameters_weight_ = (None)
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_backbone_modules_stem_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_backbone_modules_stem_modules_1_modules_conv_parameters_weight_ = (None)
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_6 = l_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_1_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_stem_modules_1_modules_bn_parameters_bias_
        ) = None
        x_8 = torch.prelu(
            x_7,
            l_self_modules_backbone_modules_stem_modules_1_modules_activate_parameters_weight_,
        )
        x_7 = l_self_modules_backbone_modules_stem_modules_1_modules_activate_parameters_weight_ = (None)
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_backbone_modules_stem_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_8 = l_self_modules_backbone_modules_stem_modules_2_modules_conv_parameters_weight_ = (None)
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_9 = l_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_stem_modules_2_modules_bn_buffers_running_var_ = (
            l_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_stem_modules_2_modules_bn_parameters_bias_
        ) = None
        x_11 = torch.prelu(
            x_10,
            l_self_modules_backbone_modules_stem_modules_2_modules_activate_parameters_weight_,
        )
        x_10 = l_self_modules_backbone_modules_stem_modules_2_modules_activate_parameters_weight_ = (None)
        cat = torch.cat([x_11, x], 1)
        x_11 = x = None
        input_1 = torch.nn.functional.batch_norm(
            cat,
            l_self_modules_backbone_modules_norm_prelu_0_modules_0_buffers_running_mean_,
            l_self_modules_backbone_modules_norm_prelu_0_modules_0_buffers_running_var_,
            l_self_modules_backbone_modules_norm_prelu_0_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_norm_prelu_0_modules_0_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        cat = (
            l_self_modules_backbone_modules_norm_prelu_0_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_norm_prelu_0_modules_0_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_norm_prelu_0_modules_0_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_norm_prelu_0_modules_0_parameters_bias_
        ) = None
        input_2 = torch.prelu(
            input_1,
            l_self_modules_backbone_modules_norm_prelu_0_modules_1_parameters_weight_,
        )
        input_1 = (
            l_self_modules_backbone_modules_norm_prelu_0_modules_1_parameters_weight_
        ) = None
        x_12 = torch.conv2d(
            input_2,
            l_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_2 = l_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_conv_parameters_weight_ = (None)
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_12 = l_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_14 = torch.prelu(
            x_13,
            l_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_13 = l_self_modules_backbone_modules_level1_modules_0_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc = torch.conv2d(
            x_14,
            l_self_modules_backbone_modules_level1_modules_0_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level1_modules_0_modules_f_loc_parameters_weight_ = (
            None
        )
        sur = torch.conv2d(
            x_14,
            l_self_modules_backbone_modules_level1_modules_0_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            64,
        )
        x_14 = l_self_modules_backbone_modules_level1_modules_0_modules_f_sur_parameters_weight_ = (None)
        joi_feat = torch.cat([loc, sur], 1)
        loc = sur = None
        joi_feat_1 = torch.nn.functional.batch_norm(
            joi_feat,
            l_self_modules_backbone_modules_level1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat = l_self_modules_backbone_modules_level1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level1_modules_0_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_level1_modules_0_modules_bn_parameters_bias_
        ) = None
        joi_feat_2 = torch.prelu(
            joi_feat_1,
            l_self_modules_backbone_modules_level1_modules_0_modules_activate_parameters_weight_,
        )
        joi_feat_1 = l_self_modules_backbone_modules_level1_modules_0_modules_activate_parameters_weight_ = (None)
        joi_feat_3 = torch.conv2d(
            joi_feat_2,
            l_self_modules_backbone_modules_level1_modules_0_modules_bottleneck_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        joi_feat_2 = l_self_modules_backbone_modules_level1_modules_0_modules_bottleneck_parameters_weight_ = (None)
        adaptive_avg_pool2d = torch.nn.functional.adaptive_avg_pool2d(joi_feat_3, 1)
        y = adaptive_avg_pool2d.view(1, 64)
        adaptive_avg_pool2d = None
        input_3 = torch._C._nn.linear(
            y,
            l_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y = l_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_4 = torch.nn.functional.relu(input_3, inplace=True)
        input_3 = None
        input_5 = torch._C._nn.linear(
            input_4,
            l_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_4 = l_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level1_modules_0_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_6 = torch.sigmoid(input_5)
        input_5 = None
        y_1 = input_6.view(1, 64, 1, 1)
        input_6 = None
        out = joi_feat_3 * y_1
        joi_feat_3 = y_1 = None
        x_15 = torch.conv2d(
            out,
            l_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_15 = l_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_17 = torch.prelu(
            x_16,
            l_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_16 = l_self_modules_backbone_modules_level1_modules_1_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_1 = torch.conv2d(
            x_17,
            l_self_modules_backbone_modules_level1_modules_1_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_backbone_modules_level1_modules_1_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_1 = torch.conv2d(
            x_17,
            l_self_modules_backbone_modules_level1_modules_1_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            32,
        )
        x_17 = l_self_modules_backbone_modules_level1_modules_1_modules_f_sur_parameters_weight_ = (None)
        joi_feat_4 = torch.cat([loc_1, sur_1], 1)
        loc_1 = sur_1 = None
        joi_feat_5 = torch.nn.functional.batch_norm(
            joi_feat_4,
            l_self_modules_backbone_modules_level1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_4 = l_self_modules_backbone_modules_level1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level1_modules_1_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_level1_modules_1_modules_bn_parameters_bias_
        ) = None
        joi_feat_6 = torch.prelu(
            joi_feat_5,
            l_self_modules_backbone_modules_level1_modules_1_modules_activate_parameters_weight_,
        )
        joi_feat_5 = l_self_modules_backbone_modules_level1_modules_1_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_1 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_6, 1)
        y_2 = adaptive_avg_pool2d_1.view(1, 64)
        adaptive_avg_pool2d_1 = None
        input_7 = torch._C._nn.linear(
            y_2,
            l_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_2 = l_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_8 = torch.nn.functional.relu(input_7, inplace=True)
        input_7 = None
        input_9 = torch._C._nn.linear(
            input_8,
            l_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_8 = l_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level1_modules_1_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_10 = torch.sigmoid(input_9)
        input_9 = None
        y_3 = input_10.view(1, 64, 1, 1)
        input_10 = None
        out_1 = joi_feat_6 * y_3
        joi_feat_6 = y_3 = None
        out_2 = out + out_1
        out_1 = None
        x_18 = torch.conv2d(
            out_2,
            l_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_18 = l_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_20 = torch.prelu(
            x_19,
            l_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_19 = l_self_modules_backbone_modules_level1_modules_2_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_2 = torch.conv2d(
            x_20,
            l_self_modules_backbone_modules_level1_modules_2_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_backbone_modules_level1_modules_2_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_2 = torch.conv2d(
            x_20,
            l_self_modules_backbone_modules_level1_modules_2_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            32,
        )
        x_20 = l_self_modules_backbone_modules_level1_modules_2_modules_f_sur_parameters_weight_ = (None)
        joi_feat_7 = torch.cat([loc_2, sur_2], 1)
        loc_2 = sur_2 = None
        joi_feat_8 = torch.nn.functional.batch_norm(
            joi_feat_7,
            l_self_modules_backbone_modules_level1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_7 = l_self_modules_backbone_modules_level1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level1_modules_2_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_level1_modules_2_modules_bn_parameters_bias_
        ) = None
        joi_feat_9 = torch.prelu(
            joi_feat_8,
            l_self_modules_backbone_modules_level1_modules_2_modules_activate_parameters_weight_,
        )
        joi_feat_8 = l_self_modules_backbone_modules_level1_modules_2_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_2 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_9, 1)
        y_4 = adaptive_avg_pool2d_2.view(1, 64)
        adaptive_avg_pool2d_2 = None
        input_11 = torch._C._nn.linear(
            y_4,
            l_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_4 = l_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_12 = torch.nn.functional.relu(input_11, inplace=True)
        input_11 = None
        input_13 = torch._C._nn.linear(
            input_12,
            l_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_12 = l_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level1_modules_2_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_14 = torch.sigmoid(input_13)
        input_13 = None
        y_5 = input_14.view(1, 64, 1, 1)
        input_14 = None
        out_3 = joi_feat_9 * y_5
        joi_feat_9 = y_5 = None
        out_4 = out_2 + out_3
        out_2 = out_3 = None
        cat_4 = torch.cat([out_4, out, x_2], 1)
        out_4 = out = x_2 = None
        input_15 = torch.nn.functional.batch_norm(
            cat_4,
            l_self_modules_backbone_modules_norm_prelu_1_modules_0_buffers_running_mean_,
            l_self_modules_backbone_modules_norm_prelu_1_modules_0_buffers_running_var_,
            l_self_modules_backbone_modules_norm_prelu_1_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_norm_prelu_1_modules_0_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        cat_4 = (
            l_self_modules_backbone_modules_norm_prelu_1_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_norm_prelu_1_modules_0_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_norm_prelu_1_modules_0_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_norm_prelu_1_modules_0_parameters_bias_
        ) = None
        input_16 = torch.prelu(
            input_15,
            l_self_modules_backbone_modules_norm_prelu_1_modules_1_parameters_weight_,
        )
        input_15 = (
            l_self_modules_backbone_modules_norm_prelu_1_modules_1_parameters_weight_
        ) = None
        x_21 = torch.conv2d(
            input_16,
            l_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_16 = l_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_conv_parameters_weight_ = (None)
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_21 = l_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_23 = torch.prelu(
            x_22,
            l_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_22 = l_self_modules_backbone_modules_level2_modules_0_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_3 = torch.conv2d(
            x_23,
            l_self_modules_backbone_modules_level2_modules_0_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_backbone_modules_level2_modules_0_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_3 = torch.conv2d(
            x_23,
            l_self_modules_backbone_modules_level2_modules_0_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            128,
        )
        x_23 = l_self_modules_backbone_modules_level2_modules_0_modules_f_sur_parameters_weight_ = (None)
        joi_feat_10 = torch.cat([loc_3, sur_3], 1)
        loc_3 = sur_3 = None
        joi_feat_11 = torch.nn.functional.batch_norm(
            joi_feat_10,
            l_self_modules_backbone_modules_level2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_10 = l_self_modules_backbone_modules_level2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_0_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_level2_modules_0_modules_bn_parameters_bias_
        ) = None
        joi_feat_12 = torch.prelu(
            joi_feat_11,
            l_self_modules_backbone_modules_level2_modules_0_modules_activate_parameters_weight_,
        )
        joi_feat_11 = l_self_modules_backbone_modules_level2_modules_0_modules_activate_parameters_weight_ = (None)
        joi_feat_13 = torch.conv2d(
            joi_feat_12,
            l_self_modules_backbone_modules_level2_modules_0_modules_bottleneck_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        joi_feat_12 = l_self_modules_backbone_modules_level2_modules_0_modules_bottleneck_parameters_weight_ = (None)
        adaptive_avg_pool2d_3 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_13, 1)
        y_6 = adaptive_avg_pool2d_3.view(1, 128)
        adaptive_avg_pool2d_3 = None
        input_17 = torch._C._nn.linear(
            y_6,
            l_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_6 = l_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_18 = torch.nn.functional.relu(input_17, inplace=True)
        input_17 = None
        input_19 = torch._C._nn.linear(
            input_18,
            l_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_18 = l_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_0_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_20 = torch.sigmoid(input_19)
        input_19 = None
        y_7 = input_20.view(1, 128, 1, 1)
        input_20 = None
        out_5 = joi_feat_13 * y_7
        joi_feat_13 = y_7 = None
        x_24 = torch.conv2d(
            out_5,
            l_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_24 = l_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_26 = torch.prelu(
            x_25,
            l_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_25 = l_self_modules_backbone_modules_level2_modules_1_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_4 = torch.conv2d(
            x_26,
            l_self_modules_backbone_modules_level2_modules_1_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_1_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_4 = torch.conv2d(
            x_26,
            l_self_modules_backbone_modules_level2_modules_1_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_26 = l_self_modules_backbone_modules_level2_modules_1_modules_f_sur_parameters_weight_ = (None)
        joi_feat_14 = torch.cat([loc_4, sur_4], 1)
        loc_4 = sur_4 = None
        joi_feat_15 = torch.nn.functional.batch_norm(
            joi_feat_14,
            l_self_modules_backbone_modules_level2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_14 = l_self_modules_backbone_modules_level2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_1_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_level2_modules_1_modules_bn_parameters_bias_
        ) = None
        joi_feat_16 = torch.prelu(
            joi_feat_15,
            l_self_modules_backbone_modules_level2_modules_1_modules_activate_parameters_weight_,
        )
        joi_feat_15 = l_self_modules_backbone_modules_level2_modules_1_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_4 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_16, 1)
        y_8 = adaptive_avg_pool2d_4.view(1, 128)
        adaptive_avg_pool2d_4 = None
        input_21 = torch._C._nn.linear(
            y_8,
            l_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_8 = l_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_22 = torch.nn.functional.relu(input_21, inplace=True)
        input_21 = None
        input_23 = torch._C._nn.linear(
            input_22,
            l_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_22 = l_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_1_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_24 = torch.sigmoid(input_23)
        input_23 = None
        y_9 = input_24.view(1, 128, 1, 1)
        input_24 = None
        out_6 = joi_feat_16 * y_9
        joi_feat_16 = y_9 = None
        out_7 = out_5 + out_6
        out_6 = None
        x_27 = torch.conv2d(
            out_7,
            l_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_28 = torch.nn.functional.batch_norm(
            x_27,
            l_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_27 = l_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_29 = torch.prelu(
            x_28,
            l_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_28 = l_self_modules_backbone_modules_level2_modules_2_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_5 = torch.conv2d(
            x_29,
            l_self_modules_backbone_modules_level2_modules_2_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_2_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_5 = torch.conv2d(
            x_29,
            l_self_modules_backbone_modules_level2_modules_2_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_29 = l_self_modules_backbone_modules_level2_modules_2_modules_f_sur_parameters_weight_ = (None)
        joi_feat_17 = torch.cat([loc_5, sur_5], 1)
        loc_5 = sur_5 = None
        joi_feat_18 = torch.nn.functional.batch_norm(
            joi_feat_17,
            l_self_modules_backbone_modules_level2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_17 = l_self_modules_backbone_modules_level2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_2_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_level2_modules_2_modules_bn_parameters_bias_
        ) = None
        joi_feat_19 = torch.prelu(
            joi_feat_18,
            l_self_modules_backbone_modules_level2_modules_2_modules_activate_parameters_weight_,
        )
        joi_feat_18 = l_self_modules_backbone_modules_level2_modules_2_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_5 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_19, 1)
        y_10 = adaptive_avg_pool2d_5.view(1, 128)
        adaptive_avg_pool2d_5 = None
        input_25 = torch._C._nn.linear(
            y_10,
            l_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_10 = l_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_26 = torch.nn.functional.relu(input_25, inplace=True)
        input_25 = None
        input_27 = torch._C._nn.linear(
            input_26,
            l_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_26 = l_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_2_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_28 = torch.sigmoid(input_27)
        input_27 = None
        y_11 = input_28.view(1, 128, 1, 1)
        input_28 = None
        out_8 = joi_feat_19 * y_11
        joi_feat_19 = y_11 = None
        out_9 = out_7 + out_8
        out_7 = out_8 = None
        x_30 = torch.conv2d(
            out_9,
            l_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_30 = l_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_32 = torch.prelu(
            x_31,
            l_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_31 = l_self_modules_backbone_modules_level2_modules_3_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_6 = torch.conv2d(
            x_32,
            l_self_modules_backbone_modules_level2_modules_3_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_3_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_6 = torch.conv2d(
            x_32,
            l_self_modules_backbone_modules_level2_modules_3_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_32 = l_self_modules_backbone_modules_level2_modules_3_modules_f_sur_parameters_weight_ = (None)
        joi_feat_20 = torch.cat([loc_6, sur_6], 1)
        loc_6 = sur_6 = None
        joi_feat_21 = torch.nn.functional.batch_norm(
            joi_feat_20,
            l_self_modules_backbone_modules_level2_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_3_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_20 = l_self_modules_backbone_modules_level2_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_3_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_3_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_level2_modules_3_modules_bn_parameters_bias_
        ) = None
        joi_feat_22 = torch.prelu(
            joi_feat_21,
            l_self_modules_backbone_modules_level2_modules_3_modules_activate_parameters_weight_,
        )
        joi_feat_21 = l_self_modules_backbone_modules_level2_modules_3_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_6 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_22, 1)
        y_12 = adaptive_avg_pool2d_6.view(1, 128)
        adaptive_avg_pool2d_6 = None
        input_29 = torch._C._nn.linear(
            y_12,
            l_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_12 = l_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_30 = torch.nn.functional.relu(input_29, inplace=True)
        input_29 = None
        input_31 = torch._C._nn.linear(
            input_30,
            l_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_30 = l_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_3_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_32 = torch.sigmoid(input_31)
        input_31 = None
        y_13 = input_32.view(1, 128, 1, 1)
        input_32 = None
        out_10 = joi_feat_22 * y_13
        joi_feat_22 = y_13 = None
        out_11 = out_9 + out_10
        out_9 = out_10 = None
        x_33 = torch.conv2d(
            out_11,
            l_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_33 = l_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_35 = torch.prelu(
            x_34,
            l_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_34 = l_self_modules_backbone_modules_level2_modules_4_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_7 = torch.conv2d(
            x_35,
            l_self_modules_backbone_modules_level2_modules_4_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_4_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_7 = torch.conv2d(
            x_35,
            l_self_modules_backbone_modules_level2_modules_4_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_35 = l_self_modules_backbone_modules_level2_modules_4_modules_f_sur_parameters_weight_ = (None)
        joi_feat_23 = torch.cat([loc_7, sur_7], 1)
        loc_7 = sur_7 = None
        joi_feat_24 = torch.nn.functional.batch_norm(
            joi_feat_23,
            l_self_modules_backbone_modules_level2_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_4_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_23 = l_self_modules_backbone_modules_level2_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_4_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_4_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_level2_modules_4_modules_bn_parameters_bias_
        ) = None
        joi_feat_25 = torch.prelu(
            joi_feat_24,
            l_self_modules_backbone_modules_level2_modules_4_modules_activate_parameters_weight_,
        )
        joi_feat_24 = l_self_modules_backbone_modules_level2_modules_4_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_7 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_25, 1)
        y_14 = adaptive_avg_pool2d_7.view(1, 128)
        adaptive_avg_pool2d_7 = None
        input_33 = torch._C._nn.linear(
            y_14,
            l_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_14 = l_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_34 = torch.nn.functional.relu(input_33, inplace=True)
        input_33 = None
        input_35 = torch._C._nn.linear(
            input_34,
            l_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_34 = l_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_4_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_36 = torch.sigmoid(input_35)
        input_35 = None
        y_15 = input_36.view(1, 128, 1, 1)
        input_36 = None
        out_12 = joi_feat_25 * y_15
        joi_feat_25 = y_15 = None
        out_13 = out_11 + out_12
        out_11 = out_12 = None
        x_36 = torch.conv2d(
            out_13,
            l_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_36 = l_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_38 = torch.prelu(
            x_37,
            l_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_37 = l_self_modules_backbone_modules_level2_modules_5_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_8 = torch.conv2d(
            x_38,
            l_self_modules_backbone_modules_level2_modules_5_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_5_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_8 = torch.conv2d(
            x_38,
            l_self_modules_backbone_modules_level2_modules_5_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_38 = l_self_modules_backbone_modules_level2_modules_5_modules_f_sur_parameters_weight_ = (None)
        joi_feat_26 = torch.cat([loc_8, sur_8], 1)
        loc_8 = sur_8 = None
        joi_feat_27 = torch.nn.functional.batch_norm(
            joi_feat_26,
            l_self_modules_backbone_modules_level2_modules_5_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_5_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_5_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_5_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_26 = l_self_modules_backbone_modules_level2_modules_5_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_5_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_5_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_level2_modules_5_modules_bn_parameters_bias_
        ) = None
        joi_feat_28 = torch.prelu(
            joi_feat_27,
            l_self_modules_backbone_modules_level2_modules_5_modules_activate_parameters_weight_,
        )
        joi_feat_27 = l_self_modules_backbone_modules_level2_modules_5_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_8 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_28, 1)
        y_16 = adaptive_avg_pool2d_8.view(1, 128)
        adaptive_avg_pool2d_8 = None
        input_37 = torch._C._nn.linear(
            y_16,
            l_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_16 = l_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_38 = torch.nn.functional.relu(input_37, inplace=True)
        input_37 = None
        input_39 = torch._C._nn.linear(
            input_38,
            l_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_38 = l_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_5_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_40 = torch.sigmoid(input_39)
        input_39 = None
        y_17 = input_40.view(1, 128, 1, 1)
        input_40 = None
        out_14 = joi_feat_28 * y_17
        joi_feat_28 = y_17 = None
        out_15 = out_13 + out_14
        out_13 = out_14 = None
        x_39 = torch.conv2d(
            out_15,
            l_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_39 = l_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_41 = torch.prelu(
            x_40,
            l_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_40 = l_self_modules_backbone_modules_level2_modules_6_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_9 = torch.conv2d(
            x_41,
            l_self_modules_backbone_modules_level2_modules_6_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_6_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_9 = torch.conv2d(
            x_41,
            l_self_modules_backbone_modules_level2_modules_6_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_41 = l_self_modules_backbone_modules_level2_modules_6_modules_f_sur_parameters_weight_ = (None)
        joi_feat_29 = torch.cat([loc_9, sur_9], 1)
        loc_9 = sur_9 = None
        joi_feat_30 = torch.nn.functional.batch_norm(
            joi_feat_29,
            l_self_modules_backbone_modules_level2_modules_6_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_6_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_6_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_6_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_29 = l_self_modules_backbone_modules_level2_modules_6_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_6_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_6_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_level2_modules_6_modules_bn_parameters_bias_
        ) = None
        joi_feat_31 = torch.prelu(
            joi_feat_30,
            l_self_modules_backbone_modules_level2_modules_6_modules_activate_parameters_weight_,
        )
        joi_feat_30 = l_self_modules_backbone_modules_level2_modules_6_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_9 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_31, 1)
        y_18 = adaptive_avg_pool2d_9.view(1, 128)
        adaptive_avg_pool2d_9 = None
        input_41 = torch._C._nn.linear(
            y_18,
            l_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_18 = l_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_42 = torch.nn.functional.relu(input_41, inplace=True)
        input_41 = None
        input_43 = torch._C._nn.linear(
            input_42,
            l_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_42 = l_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_6_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_44 = torch.sigmoid(input_43)
        input_43 = None
        y_19 = input_44.view(1, 128, 1, 1)
        input_44 = None
        out_16 = joi_feat_31 * y_19
        joi_feat_31 = y_19 = None
        out_17 = out_15 + out_16
        out_15 = out_16 = None
        x_42 = torch.conv2d(
            out_17,
            l_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_42 = l_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_44 = torch.prelu(
            x_43,
            l_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_43 = l_self_modules_backbone_modules_level2_modules_7_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_10 = torch.conv2d(
            x_44,
            l_self_modules_backbone_modules_level2_modules_7_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_7_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_10 = torch.conv2d(
            x_44,
            l_self_modules_backbone_modules_level2_modules_7_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_44 = l_self_modules_backbone_modules_level2_modules_7_modules_f_sur_parameters_weight_ = (None)
        joi_feat_32 = torch.cat([loc_10, sur_10], 1)
        loc_10 = sur_10 = None
        joi_feat_33 = torch.nn.functional.batch_norm(
            joi_feat_32,
            l_self_modules_backbone_modules_level2_modules_7_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_7_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_7_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_7_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_32 = l_self_modules_backbone_modules_level2_modules_7_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_7_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_7_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_level2_modules_7_modules_bn_parameters_bias_
        ) = None
        joi_feat_34 = torch.prelu(
            joi_feat_33,
            l_self_modules_backbone_modules_level2_modules_7_modules_activate_parameters_weight_,
        )
        joi_feat_33 = l_self_modules_backbone_modules_level2_modules_7_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_10 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_34, 1)
        y_20 = adaptive_avg_pool2d_10.view(1, 128)
        adaptive_avg_pool2d_10 = None
        input_45 = torch._C._nn.linear(
            y_20,
            l_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_20 = l_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_46 = torch.nn.functional.relu(input_45, inplace=True)
        input_45 = None
        input_47 = torch._C._nn.linear(
            input_46,
            l_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_46 = l_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_7_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_48 = torch.sigmoid(input_47)
        input_47 = None
        y_21 = input_48.view(1, 128, 1, 1)
        input_48 = None
        out_18 = joi_feat_34 * y_21
        joi_feat_34 = y_21 = None
        out_19 = out_17 + out_18
        out_17 = out_18 = None
        x_45 = torch.conv2d(
            out_19,
            l_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_45 = l_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_47 = torch.prelu(
            x_46,
            l_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_46 = l_self_modules_backbone_modules_level2_modules_8_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_11 = torch.conv2d(
            x_47,
            l_self_modules_backbone_modules_level2_modules_8_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_8_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_11 = torch.conv2d(
            x_47,
            l_self_modules_backbone_modules_level2_modules_8_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_47 = l_self_modules_backbone_modules_level2_modules_8_modules_f_sur_parameters_weight_ = (None)
        joi_feat_35 = torch.cat([loc_11, sur_11], 1)
        loc_11 = sur_11 = None
        joi_feat_36 = torch.nn.functional.batch_norm(
            joi_feat_35,
            l_self_modules_backbone_modules_level2_modules_8_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_8_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_8_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_8_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_35 = l_self_modules_backbone_modules_level2_modules_8_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_8_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_8_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_level2_modules_8_modules_bn_parameters_bias_
        ) = None
        joi_feat_37 = torch.prelu(
            joi_feat_36,
            l_self_modules_backbone_modules_level2_modules_8_modules_activate_parameters_weight_,
        )
        joi_feat_36 = l_self_modules_backbone_modules_level2_modules_8_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_11 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_37, 1)
        y_22 = adaptive_avg_pool2d_11.view(1, 128)
        adaptive_avg_pool2d_11 = None
        input_49 = torch._C._nn.linear(
            y_22,
            l_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_22 = l_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_50 = torch.nn.functional.relu(input_49, inplace=True)
        input_49 = None
        input_51 = torch._C._nn.linear(
            input_50,
            l_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_50 = l_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_8_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_52 = torch.sigmoid(input_51)
        input_51 = None
        y_23 = input_52.view(1, 128, 1, 1)
        input_52 = None
        out_20 = joi_feat_37 * y_23
        joi_feat_37 = y_23 = None
        out_21 = out_19 + out_20
        out_19 = out_20 = None
        x_48 = torch.conv2d(
            out_21,
            l_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_48 = l_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_50 = torch.prelu(
            x_49,
            l_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_49 = l_self_modules_backbone_modules_level2_modules_9_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_12 = torch.conv2d(
            x_50,
            l_self_modules_backbone_modules_level2_modules_9_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_9_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_12 = torch.conv2d(
            x_50,
            l_self_modules_backbone_modules_level2_modules_9_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_50 = l_self_modules_backbone_modules_level2_modules_9_modules_f_sur_parameters_weight_ = (None)
        joi_feat_38 = torch.cat([loc_12, sur_12], 1)
        loc_12 = sur_12 = None
        joi_feat_39 = torch.nn.functional.batch_norm(
            joi_feat_38,
            l_self_modules_backbone_modules_level2_modules_9_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_9_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_9_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_9_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_38 = l_self_modules_backbone_modules_level2_modules_9_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_9_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_9_modules_bn_parameters_weight_ = (
            l_self_modules_backbone_modules_level2_modules_9_modules_bn_parameters_bias_
        ) = None
        joi_feat_40 = torch.prelu(
            joi_feat_39,
            l_self_modules_backbone_modules_level2_modules_9_modules_activate_parameters_weight_,
        )
        joi_feat_39 = l_self_modules_backbone_modules_level2_modules_9_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_12 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_40, 1)
        y_24 = adaptive_avg_pool2d_12.view(1, 128)
        adaptive_avg_pool2d_12 = None
        input_53 = torch._C._nn.linear(
            y_24,
            l_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_24 = l_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_54 = torch.nn.functional.relu(input_53, inplace=True)
        input_53 = None
        input_55 = torch._C._nn.linear(
            input_54,
            l_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_54 = l_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_9_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_56 = torch.sigmoid(input_55)
        input_55 = None
        y_25 = input_56.view(1, 128, 1, 1)
        input_56 = None
        out_22 = joi_feat_40 * y_25
        joi_feat_40 = y_25 = None
        out_23 = out_21 + out_22
        out_21 = out_22 = None
        x_51 = torch.conv2d(
            out_23,
            l_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_51 = l_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_53 = torch.prelu(
            x_52,
            l_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_52 = l_self_modules_backbone_modules_level2_modules_10_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_13 = torch.conv2d(
            x_53,
            l_self_modules_backbone_modules_level2_modules_10_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_10_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_13 = torch.conv2d(
            x_53,
            l_self_modules_backbone_modules_level2_modules_10_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_53 = l_self_modules_backbone_modules_level2_modules_10_modules_f_sur_parameters_weight_ = (None)
        joi_feat_41 = torch.cat([loc_13, sur_13], 1)
        loc_13 = sur_13 = None
        joi_feat_42 = torch.nn.functional.batch_norm(
            joi_feat_41,
            l_self_modules_backbone_modules_level2_modules_10_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_10_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_10_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_10_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_41 = l_self_modules_backbone_modules_level2_modules_10_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_10_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_10_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_10_modules_bn_parameters_bias_ = (None)
        joi_feat_43 = torch.prelu(
            joi_feat_42,
            l_self_modules_backbone_modules_level2_modules_10_modules_activate_parameters_weight_,
        )
        joi_feat_42 = l_self_modules_backbone_modules_level2_modules_10_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_13 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_43, 1)
        y_26 = adaptive_avg_pool2d_13.view(1, 128)
        adaptive_avg_pool2d_13 = None
        input_57 = torch._C._nn.linear(
            y_26,
            l_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_26 = l_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_58 = torch.nn.functional.relu(input_57, inplace=True)
        input_57 = None
        input_59 = torch._C._nn.linear(
            input_58,
            l_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_58 = l_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_10_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_60 = torch.sigmoid(input_59)
        input_59 = None
        y_27 = input_60.view(1, 128, 1, 1)
        input_60 = None
        out_24 = joi_feat_43 * y_27
        joi_feat_43 = y_27 = None
        out_25 = out_23 + out_24
        out_23 = out_24 = None
        x_54 = torch.conv2d(
            out_25,
            l_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_54 = l_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_56 = torch.prelu(
            x_55,
            l_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_55 = l_self_modules_backbone_modules_level2_modules_11_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_14 = torch.conv2d(
            x_56,
            l_self_modules_backbone_modules_level2_modules_11_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_11_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_14 = torch.conv2d(
            x_56,
            l_self_modules_backbone_modules_level2_modules_11_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_56 = l_self_modules_backbone_modules_level2_modules_11_modules_f_sur_parameters_weight_ = (None)
        joi_feat_44 = torch.cat([loc_14, sur_14], 1)
        loc_14 = sur_14 = None
        joi_feat_45 = torch.nn.functional.batch_norm(
            joi_feat_44,
            l_self_modules_backbone_modules_level2_modules_11_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_11_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_11_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_11_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_44 = l_self_modules_backbone_modules_level2_modules_11_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_11_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_11_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_11_modules_bn_parameters_bias_ = (None)
        joi_feat_46 = torch.prelu(
            joi_feat_45,
            l_self_modules_backbone_modules_level2_modules_11_modules_activate_parameters_weight_,
        )
        joi_feat_45 = l_self_modules_backbone_modules_level2_modules_11_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_14 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_46, 1)
        y_28 = adaptive_avg_pool2d_14.view(1, 128)
        adaptive_avg_pool2d_14 = None
        input_61 = torch._C._nn.linear(
            y_28,
            l_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_28 = l_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_62 = torch.nn.functional.relu(input_61, inplace=True)
        input_61 = None
        input_63 = torch._C._nn.linear(
            input_62,
            l_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_62 = l_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_11_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_64 = torch.sigmoid(input_63)
        input_63 = None
        y_29 = input_64.view(1, 128, 1, 1)
        input_64 = None
        out_26 = joi_feat_46 * y_29
        joi_feat_46 = y_29 = None
        out_27 = out_25 + out_26
        out_25 = out_26 = None
        x_57 = torch.conv2d(
            out_27,
            l_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_58 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_57 = l_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_59 = torch.prelu(
            x_58,
            l_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_58 = l_self_modules_backbone_modules_level2_modules_12_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_15 = torch.conv2d(
            x_59,
            l_self_modules_backbone_modules_level2_modules_12_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_12_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_15 = torch.conv2d(
            x_59,
            l_self_modules_backbone_modules_level2_modules_12_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_59 = l_self_modules_backbone_modules_level2_modules_12_modules_f_sur_parameters_weight_ = (None)
        joi_feat_47 = torch.cat([loc_15, sur_15], 1)
        loc_15 = sur_15 = None
        joi_feat_48 = torch.nn.functional.batch_norm(
            joi_feat_47,
            l_self_modules_backbone_modules_level2_modules_12_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_12_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_12_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_12_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_47 = l_self_modules_backbone_modules_level2_modules_12_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_12_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_12_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_12_modules_bn_parameters_bias_ = (None)
        joi_feat_49 = torch.prelu(
            joi_feat_48,
            l_self_modules_backbone_modules_level2_modules_12_modules_activate_parameters_weight_,
        )
        joi_feat_48 = l_self_modules_backbone_modules_level2_modules_12_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_15 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_49, 1)
        y_30 = adaptive_avg_pool2d_15.view(1, 128)
        adaptive_avg_pool2d_15 = None
        input_65 = torch._C._nn.linear(
            y_30,
            l_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_30 = l_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_66 = torch.nn.functional.relu(input_65, inplace=True)
        input_65 = None
        input_67 = torch._C._nn.linear(
            input_66,
            l_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_66 = l_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_12_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_68 = torch.sigmoid(input_67)
        input_67 = None
        y_31 = input_68.view(1, 128, 1, 1)
        input_68 = None
        out_28 = joi_feat_49 * y_31
        joi_feat_49 = y_31 = None
        out_29 = out_27 + out_28
        out_27 = out_28 = None
        x_60 = torch.conv2d(
            out_29,
            l_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_60 = l_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_62 = torch.prelu(
            x_61,
            l_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_61 = l_self_modules_backbone_modules_level2_modules_13_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_16 = torch.conv2d(
            x_62,
            l_self_modules_backbone_modules_level2_modules_13_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_13_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_16 = torch.conv2d(
            x_62,
            l_self_modules_backbone_modules_level2_modules_13_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_62 = l_self_modules_backbone_modules_level2_modules_13_modules_f_sur_parameters_weight_ = (None)
        joi_feat_50 = torch.cat([loc_16, sur_16], 1)
        loc_16 = sur_16 = None
        joi_feat_51 = torch.nn.functional.batch_norm(
            joi_feat_50,
            l_self_modules_backbone_modules_level2_modules_13_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_13_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_13_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_13_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_50 = l_self_modules_backbone_modules_level2_modules_13_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_13_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_13_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_13_modules_bn_parameters_bias_ = (None)
        joi_feat_52 = torch.prelu(
            joi_feat_51,
            l_self_modules_backbone_modules_level2_modules_13_modules_activate_parameters_weight_,
        )
        joi_feat_51 = l_self_modules_backbone_modules_level2_modules_13_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_16 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_52, 1)
        y_32 = adaptive_avg_pool2d_16.view(1, 128)
        adaptive_avg_pool2d_16 = None
        input_69 = torch._C._nn.linear(
            y_32,
            l_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_32 = l_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_70 = torch.nn.functional.relu(input_69, inplace=True)
        input_69 = None
        input_71 = torch._C._nn.linear(
            input_70,
            l_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_70 = l_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_13_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_72 = torch.sigmoid(input_71)
        input_71 = None
        y_33 = input_72.view(1, 128, 1, 1)
        input_72 = None
        out_30 = joi_feat_52 * y_33
        joi_feat_52 = y_33 = None
        out_31 = out_29 + out_30
        out_29 = out_30 = None
        x_63 = torch.conv2d(
            out_31,
            l_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_63 = l_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_65 = torch.prelu(
            x_64,
            l_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_64 = l_self_modules_backbone_modules_level2_modules_14_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_17 = torch.conv2d(
            x_65,
            l_self_modules_backbone_modules_level2_modules_14_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_14_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_17 = torch.conv2d(
            x_65,
            l_self_modules_backbone_modules_level2_modules_14_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_65 = l_self_modules_backbone_modules_level2_modules_14_modules_f_sur_parameters_weight_ = (None)
        joi_feat_53 = torch.cat([loc_17, sur_17], 1)
        loc_17 = sur_17 = None
        joi_feat_54 = torch.nn.functional.batch_norm(
            joi_feat_53,
            l_self_modules_backbone_modules_level2_modules_14_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_14_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_14_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_14_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_53 = l_self_modules_backbone_modules_level2_modules_14_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_14_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_14_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_14_modules_bn_parameters_bias_ = (None)
        joi_feat_55 = torch.prelu(
            joi_feat_54,
            l_self_modules_backbone_modules_level2_modules_14_modules_activate_parameters_weight_,
        )
        joi_feat_54 = l_self_modules_backbone_modules_level2_modules_14_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_17 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_55, 1)
        y_34 = adaptive_avg_pool2d_17.view(1, 128)
        adaptive_avg_pool2d_17 = None
        input_73 = torch._C._nn.linear(
            y_34,
            l_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_34 = l_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_74 = torch.nn.functional.relu(input_73, inplace=True)
        input_73 = None
        input_75 = torch._C._nn.linear(
            input_74,
            l_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_74 = l_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_14_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_76 = torch.sigmoid(input_75)
        input_75 = None
        y_35 = input_76.view(1, 128, 1, 1)
        input_76 = None
        out_32 = joi_feat_55 * y_35
        joi_feat_55 = y_35 = None
        out_33 = out_31 + out_32
        out_31 = out_32 = None
        x_66 = torch.conv2d(
            out_33,
            l_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_66 = l_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_68 = torch.prelu(
            x_67,
            l_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_67 = l_self_modules_backbone_modules_level2_modules_15_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_18 = torch.conv2d(
            x_68,
            l_self_modules_backbone_modules_level2_modules_15_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_15_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_18 = torch.conv2d(
            x_68,
            l_self_modules_backbone_modules_level2_modules_15_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_68 = l_self_modules_backbone_modules_level2_modules_15_modules_f_sur_parameters_weight_ = (None)
        joi_feat_56 = torch.cat([loc_18, sur_18], 1)
        loc_18 = sur_18 = None
        joi_feat_57 = torch.nn.functional.batch_norm(
            joi_feat_56,
            l_self_modules_backbone_modules_level2_modules_15_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_15_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_15_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_15_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_56 = l_self_modules_backbone_modules_level2_modules_15_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_15_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_15_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_15_modules_bn_parameters_bias_ = (None)
        joi_feat_58 = torch.prelu(
            joi_feat_57,
            l_self_modules_backbone_modules_level2_modules_15_modules_activate_parameters_weight_,
        )
        joi_feat_57 = l_self_modules_backbone_modules_level2_modules_15_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_18 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_58, 1)
        y_36 = adaptive_avg_pool2d_18.view(1, 128)
        adaptive_avg_pool2d_18 = None
        input_77 = torch._C._nn.linear(
            y_36,
            l_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_36 = l_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_78 = torch.nn.functional.relu(input_77, inplace=True)
        input_77 = None
        input_79 = torch._C._nn.linear(
            input_78,
            l_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_78 = l_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_15_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_80 = torch.sigmoid(input_79)
        input_79 = None
        y_37 = input_80.view(1, 128, 1, 1)
        input_80 = None
        out_34 = joi_feat_58 * y_37
        joi_feat_58 = y_37 = None
        out_35 = out_33 + out_34
        out_33 = out_34 = None
        x_69 = torch.conv2d(
            out_35,
            l_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_69 = l_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_71 = torch.prelu(
            x_70,
            l_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_70 = l_self_modules_backbone_modules_level2_modules_16_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_19 = torch.conv2d(
            x_71,
            l_self_modules_backbone_modules_level2_modules_16_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_16_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_19 = torch.conv2d(
            x_71,
            l_self_modules_backbone_modules_level2_modules_16_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_71 = l_self_modules_backbone_modules_level2_modules_16_modules_f_sur_parameters_weight_ = (None)
        joi_feat_59 = torch.cat([loc_19, sur_19], 1)
        loc_19 = sur_19 = None
        joi_feat_60 = torch.nn.functional.batch_norm(
            joi_feat_59,
            l_self_modules_backbone_modules_level2_modules_16_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_16_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_16_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_16_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_59 = l_self_modules_backbone_modules_level2_modules_16_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_16_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_16_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_16_modules_bn_parameters_bias_ = (None)
        joi_feat_61 = torch.prelu(
            joi_feat_60,
            l_self_modules_backbone_modules_level2_modules_16_modules_activate_parameters_weight_,
        )
        joi_feat_60 = l_self_modules_backbone_modules_level2_modules_16_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_19 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_61, 1)
        y_38 = adaptive_avg_pool2d_19.view(1, 128)
        adaptive_avg_pool2d_19 = None
        input_81 = torch._C._nn.linear(
            y_38,
            l_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_38 = l_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_82 = torch.nn.functional.relu(input_81, inplace=True)
        input_81 = None
        input_83 = torch._C._nn.linear(
            input_82,
            l_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_82 = l_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_16_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_84 = torch.sigmoid(input_83)
        input_83 = None
        y_39 = input_84.view(1, 128, 1, 1)
        input_84 = None
        out_36 = joi_feat_61 * y_39
        joi_feat_61 = y_39 = None
        out_37 = out_35 + out_36
        out_35 = out_36 = None
        x_72 = torch.conv2d(
            out_37,
            l_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_72 = l_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_74 = torch.prelu(
            x_73,
            l_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_73 = l_self_modules_backbone_modules_level2_modules_17_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_20 = torch.conv2d(
            x_74,
            l_self_modules_backbone_modules_level2_modules_17_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_17_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_20 = torch.conv2d(
            x_74,
            l_self_modules_backbone_modules_level2_modules_17_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_74 = l_self_modules_backbone_modules_level2_modules_17_modules_f_sur_parameters_weight_ = (None)
        joi_feat_62 = torch.cat([loc_20, sur_20], 1)
        loc_20 = sur_20 = None
        joi_feat_63 = torch.nn.functional.batch_norm(
            joi_feat_62,
            l_self_modules_backbone_modules_level2_modules_17_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_17_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_17_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_17_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_62 = l_self_modules_backbone_modules_level2_modules_17_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_17_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_17_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_17_modules_bn_parameters_bias_ = (None)
        joi_feat_64 = torch.prelu(
            joi_feat_63,
            l_self_modules_backbone_modules_level2_modules_17_modules_activate_parameters_weight_,
        )
        joi_feat_63 = l_self_modules_backbone_modules_level2_modules_17_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_20 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_64, 1)
        y_40 = adaptive_avg_pool2d_20.view(1, 128)
        adaptive_avg_pool2d_20 = None
        input_85 = torch._C._nn.linear(
            y_40,
            l_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_40 = l_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_86 = torch.nn.functional.relu(input_85, inplace=True)
        input_85 = None
        input_87 = torch._C._nn.linear(
            input_86,
            l_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_86 = l_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_17_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_88 = torch.sigmoid(input_87)
        input_87 = None
        y_41 = input_88.view(1, 128, 1, 1)
        input_88 = None
        out_38 = joi_feat_64 * y_41
        joi_feat_64 = y_41 = None
        out_39 = out_37 + out_38
        out_37 = out_38 = None
        x_75 = torch.conv2d(
            out_39,
            l_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_75 = l_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_77 = torch.prelu(
            x_76,
            l_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_76 = l_self_modules_backbone_modules_level2_modules_18_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_21 = torch.conv2d(
            x_77,
            l_self_modules_backbone_modules_level2_modules_18_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_18_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_21 = torch.conv2d(
            x_77,
            l_self_modules_backbone_modules_level2_modules_18_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_77 = l_self_modules_backbone_modules_level2_modules_18_modules_f_sur_parameters_weight_ = (None)
        joi_feat_65 = torch.cat([loc_21, sur_21], 1)
        loc_21 = sur_21 = None
        joi_feat_66 = torch.nn.functional.batch_norm(
            joi_feat_65,
            l_self_modules_backbone_modules_level2_modules_18_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_18_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_18_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_18_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_65 = l_self_modules_backbone_modules_level2_modules_18_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_18_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_18_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_18_modules_bn_parameters_bias_ = (None)
        joi_feat_67 = torch.prelu(
            joi_feat_66,
            l_self_modules_backbone_modules_level2_modules_18_modules_activate_parameters_weight_,
        )
        joi_feat_66 = l_self_modules_backbone_modules_level2_modules_18_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_21 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_67, 1)
        y_42 = adaptive_avg_pool2d_21.view(1, 128)
        adaptive_avg_pool2d_21 = None
        input_89 = torch._C._nn.linear(
            y_42,
            l_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_42 = l_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_90 = torch.nn.functional.relu(input_89, inplace=True)
        input_89 = None
        input_91 = torch._C._nn.linear(
            input_90,
            l_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_90 = l_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_18_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_92 = torch.sigmoid(input_91)
        input_91 = None
        y_43 = input_92.view(1, 128, 1, 1)
        input_92 = None
        out_40 = joi_feat_67 * y_43
        joi_feat_67 = y_43 = None
        out_41 = out_39 + out_40
        out_39 = out_40 = None
        x_78 = torch.conv2d(
            out_41,
            l_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_79 = torch.nn.functional.batch_norm(
            x_78,
            l_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_78 = l_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_80 = torch.prelu(
            x_79,
            l_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_79 = l_self_modules_backbone_modules_level2_modules_19_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_22 = torch.conv2d(
            x_80,
            l_self_modules_backbone_modules_level2_modules_19_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_19_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_22 = torch.conv2d(
            x_80,
            l_self_modules_backbone_modules_level2_modules_19_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_80 = l_self_modules_backbone_modules_level2_modules_19_modules_f_sur_parameters_weight_ = (None)
        joi_feat_68 = torch.cat([loc_22, sur_22], 1)
        loc_22 = sur_22 = None
        joi_feat_69 = torch.nn.functional.batch_norm(
            joi_feat_68,
            l_self_modules_backbone_modules_level2_modules_19_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_19_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_19_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_19_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_68 = l_self_modules_backbone_modules_level2_modules_19_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_19_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_19_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_19_modules_bn_parameters_bias_ = (None)
        joi_feat_70 = torch.prelu(
            joi_feat_69,
            l_self_modules_backbone_modules_level2_modules_19_modules_activate_parameters_weight_,
        )
        joi_feat_69 = l_self_modules_backbone_modules_level2_modules_19_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_22 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_70, 1)
        y_44 = adaptive_avg_pool2d_22.view(1, 128)
        adaptive_avg_pool2d_22 = None
        input_93 = torch._C._nn.linear(
            y_44,
            l_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_44 = l_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_94 = torch.nn.functional.relu(input_93, inplace=True)
        input_93 = None
        input_95 = torch._C._nn.linear(
            input_94,
            l_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_94 = l_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_19_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_96 = torch.sigmoid(input_95)
        input_95 = None
        y_45 = input_96.view(1, 128, 1, 1)
        input_96 = None
        out_42 = joi_feat_70 * y_45
        joi_feat_70 = y_45 = None
        out_43 = out_41 + out_42
        out_41 = out_42 = None
        x_81 = torch.conv2d(
            out_43,
            l_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_82 = torch.nn.functional.batch_norm(
            x_81,
            l_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_81 = l_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_bn_parameters_bias_ = (None)
        x_83 = torch.prelu(
            x_82,
            l_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_activate_parameters_weight_,
        )
        x_82 = l_self_modules_backbone_modules_level2_modules_20_modules_conv1x1_modules_activate_parameters_weight_ = (None)
        loc_23 = torch.conv2d(
            x_83,
            l_self_modules_backbone_modules_level2_modules_20_modules_f_loc_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_backbone_modules_level2_modules_20_modules_f_loc_parameters_weight_ = (
            None
        )
        sur_23 = torch.conv2d(
            x_83,
            l_self_modules_backbone_modules_level2_modules_20_modules_f_sur_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (4, 4),
            64,
        )
        x_83 = l_self_modules_backbone_modules_level2_modules_20_modules_f_sur_parameters_weight_ = (None)
        joi_feat_71 = torch.cat([loc_23, sur_23], 1)
        loc_23 = sur_23 = None
        joi_feat_72 = torch.nn.functional.batch_norm(
            joi_feat_71,
            l_self_modules_backbone_modules_level2_modules_20_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_level2_modules_20_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_level2_modules_20_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_20_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        joi_feat_71 = l_self_modules_backbone_modules_level2_modules_20_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_level2_modules_20_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_level2_modules_20_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_20_modules_bn_parameters_bias_ = (None)
        joi_feat_73 = torch.prelu(
            joi_feat_72,
            l_self_modules_backbone_modules_level2_modules_20_modules_activate_parameters_weight_,
        )
        joi_feat_72 = l_self_modules_backbone_modules_level2_modules_20_modules_activate_parameters_weight_ = (None)
        adaptive_avg_pool2d_23 = torch.nn.functional.adaptive_avg_pool2d(joi_feat_73, 1)
        y_46 = adaptive_avg_pool2d_23.view(1, 128)
        adaptive_avg_pool2d_23 = None
        input_97 = torch._C._nn.linear(
            y_46,
            l_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_0_parameters_bias_,
        )
        y_46 = l_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_0_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_0_parameters_bias_ = (None)
        input_98 = torch.nn.functional.relu(input_97, inplace=True)
        input_97 = None
        input_99 = torch._C._nn.linear(
            input_98,
            l_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_2_parameters_bias_,
        )
        input_98 = l_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_2_parameters_weight_ = l_self_modules_backbone_modules_level2_modules_20_modules_f_glo_modules_fc_modules_2_parameters_bias_ = (None)
        input_100 = torch.sigmoid(input_99)
        input_99 = None
        y_47 = input_100.view(1, 128, 1, 1)
        input_100 = None
        out_44 = joi_feat_73 * y_47
        joi_feat_73 = y_47 = None
        out_45 = out_43 + out_44
        out_43 = out_44 = None
        cat_26 = torch.cat([out_5, out_45], 1)
        out_5 = out_45 = None
        input_101 = torch.nn.functional.batch_norm(
            cat_26,
            l_self_modules_backbone_modules_norm_prelu_2_modules_0_buffers_running_mean_,
            l_self_modules_backbone_modules_norm_prelu_2_modules_0_buffers_running_var_,
            l_self_modules_backbone_modules_norm_prelu_2_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_norm_prelu_2_modules_0_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        cat_26 = (
            l_self_modules_backbone_modules_norm_prelu_2_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_norm_prelu_2_modules_0_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_norm_prelu_2_modules_0_parameters_weight_
        ) = (
            l_self_modules_backbone_modules_norm_prelu_2_modules_0_parameters_bias_
        ) = None
        input_102 = torch.prelu(
            input_101,
            l_self_modules_backbone_modules_norm_prelu_2_modules_1_parameters_weight_,
        )
        input_101 = (
            l_self_modules_backbone_modules_norm_prelu_2_modules_1_parameters_weight_
        ) = None
        output = torch.conv2d(
            input_102,
            l_self_modules_decode_head_modules_conv_seg_parameters_weight_,
            l_self_modules_decode_head_modules_conv_seg_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_102 = (
            l_self_modules_decode_head_modules_conv_seg_parameters_weight_
        ) = l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = None
        return (output,)
