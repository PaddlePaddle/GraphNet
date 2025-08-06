import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_14_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_14_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_15_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_15_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_conv_scale_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv_scale_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_conv_scale_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_conv_scale_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_conv_scale_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_conv_scale_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_conv_scale_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_conv_scale_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_conv_scale_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_conv_scale_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stem_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_0_modules_2_modules_identity_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_identity_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_2_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_0_modules_3_modules_identity_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_identity_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_3_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_2_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_2_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_3_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_3_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_4_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_4_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_5_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_5_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_6_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_6_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_7_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_7_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_8_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_8_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_9_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_9_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_10_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_10_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_11_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_11_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_12_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_12_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_13_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_13_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_14_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_14_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_14_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_14_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_14_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_14_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_14_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_14_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_15_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_15_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_15_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_15_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_15_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_15_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_15_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_15_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_2_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_2_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_3_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_3_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_4_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_4_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_5_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_5_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_6_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_6_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_7_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_7_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_8_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_8_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_9_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_9_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_10_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_10_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_11_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_11_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_12_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_12_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_13_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_13_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_14_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_14_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_15_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_15_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_16_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_16_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_17_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_17_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_18_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_18_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_19_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_19_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stem_modules_conv_scale_modules_conv_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_stem_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_stem_modules_conv_scale_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_conv_scale_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_conv_scale_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_conv_scale_modules_bn_parameters_bias_ = None
        x_2 = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_3 = torch.nn.functional.batch_norm(
            x_2,
            l_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_2 = l_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = (
            l_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        ) = (
            l_self_modules_stem_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        ) = None
        x_1 += x_3
        out = x_1
        x_1 = x_3 = None
        out += 0
        out_1 = out
        out = None
        x_4 = torch.nn.functional.relu(out_1, inplace=True)
        out_1 = None
        x_5 = torch.conv2d(
            x_4,
            l_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            64,
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_6 = torch.nn.functional.batch_norm(
            x_5,
            l_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_5 = l_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_7 = torch.conv2d(
            x_4,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            64,
        )
        x_4 = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_6 += x_8
        out_2 = x_6
        x_6 = x_8 = None
        out_2 += 0
        out_3 = out_2
        out_2 = None
        input_1 = torch.nn.functional.relu(out_3, inplace=True)
        out_3 = None
        x_9 = torch.conv2d(
            input_1,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_1 = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_4 = 0 + x_10
        x_10 = None
        out_4 += 0
        out_5 = out_4
        out_4 = None
        input_2 = torch.nn.functional.relu(out_5, inplace=True)
        out_5 = None
        x_11 = torch.nn.functional.batch_norm(
            input_2,
            l_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_2_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_0_modules_2_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_0_modules_2_modules_identity_parameters_bias_
        ) = None
        x_12 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_12 = l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_14 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        input_2 = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_14 = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_13 += x_15
        out_6 = x_13
        x_13 = x_15 = None
        out_6 += x_11
        out_7 = out_6
        out_6 = x_11 = None
        input_3 = torch.nn.functional.relu(out_7, inplace=True)
        out_7 = None
        x_16 = torch.nn.functional.batch_norm(
            input_3,
            l_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_3_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_0_modules_3_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_0_modules_3_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_0_modules_3_modules_identity_parameters_bias_
        ) = None
        x_17 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_3 = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_18 = torch.nn.functional.batch_norm(
            x_17,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_17 = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_8 = 0 + x_18
        x_18 = None
        out_8 += x_16
        out_9 = out_8
        out_8 = x_16 = None
        input_4 = torch.nn.functional.relu(out_9, inplace=True)
        out_9 = None
        x_19 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_21 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            128,
        )
        input_4 = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_21 = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_20 += x_22
        out_10 = x_20
        x_20 = x_22 = None
        out_10 += 0
        out_11 = out_10
        out_10 = None
        input_5 = torch.nn.functional.relu(out_11, inplace=True)
        out_11 = None
        x_23 = torch.conv2d(
            input_5,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_5 = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_12 = 0 + x_24
        x_24 = None
        out_12 += 0
        out_13 = out_12
        out_12 = None
        input_6 = torch.nn.functional.relu(out_13, inplace=True)
        out_13 = None
        x_25 = torch.nn.functional.batch_norm(
            input_6,
            l_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_2_modules_identity_parameters_bias_
        ) = None
        x_26 = torch.conv2d(
            input_6,
            l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            320,
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_28 = torch.conv2d(
            input_6,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        input_6 = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_27 += x_29
        out_14 = x_27
        x_27 = x_29 = None
        out_14 += x_25
        out_15 = out_14
        out_14 = x_25 = None
        input_7 = torch.nn.functional.relu(out_15, inplace=True)
        out_15 = None
        x_30 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_3_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_3_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_3_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_3_modules_identity_parameters_bias_
        ) = None
        x_31 = torch.conv2d(
            input_7,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_7 = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_16 = 0 + x_32
        x_32 = None
        out_16 += x_30
        out_17 = out_16
        out_16 = x_30 = None
        input_8 = torch.nn.functional.relu(out_17, inplace=True)
        out_17 = None
        x_33 = torch.nn.functional.batch_norm(
            input_8,
            l_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_4_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_4_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_4_modules_identity_parameters_bias_
        ) = None
        x_34 = torch.conv2d(
            input_8,
            l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            320,
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_36 = torch.conv2d(
            input_8,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        input_8 = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_35 += x_37
        out_18 = x_35
        x_35 = x_37 = None
        out_18 += x_33
        out_19 = out_18
        out_18 = x_33 = None
        input_9 = torch.nn.functional.relu(out_19, inplace=True)
        out_19 = None
        x_38 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_5_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_5_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_5_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_5_modules_identity_parameters_bias_
        ) = None
        x_39 = torch.conv2d(
            input_9,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_9 = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_39 = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_20 = 0 + x_40
        x_40 = None
        out_20 += x_38
        out_21 = out_20
        out_20 = x_38 = None
        input_10 = torch.nn.functional.relu(out_21, inplace=True)
        out_21 = None
        x_41 = torch.nn.functional.batch_norm(
            input_10,
            l_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_6_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_6_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_6_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_6_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_6_modules_identity_parameters_bias_
        ) = None
        x_42 = torch.conv2d(
            input_10,
            l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            320,
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_44 = torch.conv2d(
            input_10,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        input_10 = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_45 = torch.nn.functional.batch_norm(
            x_44,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_44 = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_43 += x_45
        out_22 = x_43
        x_43 = x_45 = None
        out_22 += x_41
        out_23 = out_22
        out_22 = x_41 = None
        input_11 = torch.nn.functional.relu(out_23, inplace=True)
        out_23 = None
        x_46 = torch.nn.functional.batch_norm(
            input_11,
            l_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_7_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_7_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_7_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_7_modules_identity_parameters_bias_
        ) = None
        x_47 = torch.conv2d(
            input_11,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_11 = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_24 = 0 + x_48
        x_48 = None
        out_24 += x_46
        out_25 = out_24
        out_24 = x_46 = None
        input_12 = torch.nn.functional.relu(out_25, inplace=True)
        out_25 = None
        x_49 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_8_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_8_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_8_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_8_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_8_modules_identity_parameters_bias_
        ) = None
        x_50 = torch.conv2d(
            input_12,
            l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            320,
        )
        l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_52 = torch.conv2d(
            input_12,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        input_12 = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_51 += x_53
        out_26 = x_51
        x_51 = x_53 = None
        out_26 += x_49
        out_27 = out_26
        out_26 = x_49 = None
        input_13 = torch.nn.functional.relu(out_27, inplace=True)
        out_27 = None
        x_54 = torch.nn.functional.batch_norm(
            input_13,
            l_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_9_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_9_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_9_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_9_modules_identity_parameters_bias_
        ) = None
        x_55 = torch.conv2d(
            input_13,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_13 = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_55 = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_28 = 0 + x_56
        x_56 = None
        out_28 += x_54
        out_29 = out_28
        out_28 = x_54 = None
        input_14 = torch.nn.functional.relu(out_29, inplace=True)
        out_29 = None
        x_57 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_10_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_10_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_10_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_10_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_10_modules_identity_parameters_bias_
        ) = None
        x_58 = torch.conv2d(
            input_14,
            l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            320,
        )
        l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_60 = torch.conv2d(
            input_14,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        input_14 = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_60 = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_59 += x_61
        out_30 = x_59
        x_59 = x_61 = None
        out_30 += x_57
        out_31 = out_30
        out_30 = x_57 = None
        input_15 = torch.nn.functional.relu(out_31, inplace=True)
        out_31 = None
        x_62 = torch.nn.functional.batch_norm(
            input_15,
            l_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_11_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_11_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_11_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_11_modules_identity_parameters_bias_
        ) = None
        x_63 = torch.conv2d(
            input_15,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_15 = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_32 = 0 + x_64
        x_64 = None
        out_32 += x_62
        out_33 = out_32
        out_32 = x_62 = None
        input_16 = torch.nn.functional.relu(out_33, inplace=True)
        out_33 = None
        x_65 = torch.nn.functional.batch_norm(
            input_16,
            l_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_12_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_12_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_12_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_12_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_12_modules_identity_parameters_bias_
        ) = None
        x_66 = torch.conv2d(
            input_16,
            l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            320,
        )
        l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_68 = torch.conv2d(
            input_16,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        input_16 = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_67 += x_69
        out_34 = x_67
        x_67 = x_69 = None
        out_34 += x_65
        out_35 = out_34
        out_34 = x_65 = None
        input_17 = torch.nn.functional.relu(out_35, inplace=True)
        out_35 = None
        x_70 = torch.nn.functional.batch_norm(
            input_17,
            l_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_13_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_13_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_13_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_13_modules_identity_parameters_bias_
        ) = None
        x_71 = torch.conv2d(
            input_17,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_17 = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_72 = torch.nn.functional.batch_norm(
            x_71,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_71 = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_36 = 0 + x_72
        x_72 = None
        out_36 += x_70
        out_37 = out_36
        out_36 = x_70 = None
        input_18 = torch.nn.functional.relu(out_37, inplace=True)
        out_37 = None
        x_73 = torch.nn.functional.batch_norm(
            input_18,
            l_self_modules_stages_modules_1_modules_14_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_14_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_14_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_14_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_14_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_14_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_14_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_14_modules_identity_parameters_bias_
        ) = None
        x_74 = torch.conv2d(
            input_18,
            l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            320,
        )
        l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_75 = torch.nn.functional.batch_norm(
            x_74,
            l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_74 = l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_76 = torch.conv2d(
            input_18,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            320,
        )
        input_18 = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_75 += x_77
        out_38 = x_75
        x_75 = x_77 = None
        out_38 += x_73
        out_39 = out_38
        out_38 = x_73 = None
        input_19 = torch.nn.functional.relu(out_39, inplace=True)
        out_39 = None
        x_78 = torch.nn.functional.batch_norm(
            input_19,
            l_self_modules_stages_modules_1_modules_15_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_15_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_15_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_15_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_15_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_15_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_15_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_15_modules_identity_parameters_bias_
        ) = None
        x_79 = torch.conv2d(
            input_19,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_19 = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_40 = 0 + x_80
        x_80 = None
        out_40 += x_78
        out_41 = out_40
        out_40 = x_78 = None
        input_20 = torch.nn.functional.relu(out_41, inplace=True)
        out_41 = None
        x_81 = torch.conv2d(
            input_20,
            l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            320,
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_82 = torch.nn.functional.batch_norm(
            x_81,
            l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_81 = l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_83 = torch.conv2d(
            input_20,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            320,
        )
        input_20 = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_82 += x_84
        out_42 = x_82
        x_82 = x_84 = None
        out_42 += 0
        out_43 = out_42
        out_42 = None
        input_21 = torch.nn.functional.relu(out_43, inplace=True)
        out_43 = None
        x_85 = torch.conv2d(
            input_21,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_21 = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_85 = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_44 = 0 + x_86
        x_86 = None
        out_44 += 0
        out_45 = out_44
        out_44 = None
        input_22 = torch.nn.functional.relu(out_45, inplace=True)
        out_45 = None
        x_87 = torch.nn.functional.batch_norm(
            input_22,
            l_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_2_modules_identity_parameters_bias_
        ) = None
        x_88 = torch.conv2d(
            input_22,
            l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            768,
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_90 = torch.conv2d(
            input_22,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_22 = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_91 = torch.nn.functional.batch_norm(
            x_90,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_90 = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_89 += x_91
        out_46 = x_89
        x_89 = x_91 = None
        out_46 += x_87
        out_47 = out_46
        out_46 = x_87 = None
        input_23 = torch.nn.functional.relu(out_47, inplace=True)
        out_47 = None
        x_92 = torch.nn.functional.batch_norm(
            input_23,
            l_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_3_modules_identity_parameters_bias_
        ) = None
        x_93 = torch.conv2d(
            input_23,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_23 = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_94 = torch.nn.functional.batch_norm(
            x_93,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_93 = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_48 = 0 + x_94
        x_94 = None
        out_48 += x_92
        out_49 = out_48
        out_48 = x_92 = None
        input_24 = torch.nn.functional.relu(out_49, inplace=True)
        out_49 = None
        x_95 = torch.nn.functional.batch_norm(
            input_24,
            l_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_4_modules_identity_parameters_bias_
        ) = None
        x_96 = torch.conv2d(
            input_24,
            l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            768,
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_96 = l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_98 = torch.conv2d(
            input_24,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_24 = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_98 = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_97 += x_99
        out_50 = x_97
        x_97 = x_99 = None
        out_50 += x_95
        out_51 = out_50
        out_50 = x_95 = None
        input_25 = torch.nn.functional.relu(out_51, inplace=True)
        out_51 = None
        x_100 = torch.nn.functional.batch_norm(
            input_25,
            l_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_5_modules_identity_parameters_bias_
        ) = None
        x_101 = torch.conv2d(
            input_25,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_25 = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_52 = 0 + x_102
        x_102 = None
        out_52 += x_100
        out_53 = out_52
        out_52 = x_100 = None
        input_26 = torch.nn.functional.relu(out_53, inplace=True)
        out_53 = None
        x_103 = torch.nn.functional.batch_norm(
            input_26,
            l_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_6_modules_identity_parameters_bias_
        ) = None
        x_104 = torch.conv2d(
            input_26,
            l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            768,
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_104 = l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_106 = torch.conv2d(
            input_26,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_26 = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_107 = torch.nn.functional.batch_norm(
            x_106,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_106 = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_105 += x_107
        out_54 = x_105
        x_105 = x_107 = None
        out_54 += x_103
        out_55 = out_54
        out_54 = x_103 = None
        input_27 = torch.nn.functional.relu(out_55, inplace=True)
        out_55 = None
        x_108 = torch.nn.functional.batch_norm(
            input_27,
            l_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_7_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_7_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_7_modules_identity_parameters_bias_
        ) = None
        x_109 = torch.conv2d(
            input_27,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_27 = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_110 = torch.nn.functional.batch_norm(
            x_109,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_109 = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_56 = 0 + x_110
        x_110 = None
        out_56 += x_108
        out_57 = out_56
        out_56 = x_108 = None
        input_28 = torch.nn.functional.relu(out_57, inplace=True)
        out_57 = None
        x_111 = torch.nn.functional.batch_norm(
            input_28,
            l_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_8_modules_identity_parameters_bias_
        ) = None
        x_112 = torch.conv2d(
            input_28,
            l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            768,
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_112 = l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_114 = torch.conv2d(
            input_28,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_28 = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_115 = torch.nn.functional.batch_norm(
            x_114,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_114 = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_113 += x_115
        out_58 = x_113
        x_113 = x_115 = None
        out_58 += x_111
        out_59 = out_58
        out_58 = x_111 = None
        input_29 = torch.nn.functional.relu(out_59, inplace=True)
        out_59 = None
        x_116 = torch.nn.functional.batch_norm(
            input_29,
            l_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_9_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_9_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_9_modules_identity_parameters_bias_
        ) = None
        x_117 = torch.conv2d(
            input_29,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_29 = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_117 = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_60 = 0 + x_118
        x_118 = None
        out_60 += x_116
        out_61 = out_60
        out_60 = x_116 = None
        input_30 = torch.nn.functional.relu(out_61, inplace=True)
        out_61 = None
        x_119 = torch.nn.functional.batch_norm(
            input_30,
            l_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_10_modules_identity_parameters_bias_
        ) = None
        x_120 = torch.conv2d(
            input_30,
            l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            768,
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_122 = torch.conv2d(
            input_30,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_30 = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_123 = torch.nn.functional.batch_norm(
            x_122,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_122 = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_121 += x_123
        out_62 = x_121
        x_121 = x_123 = None
        out_62 += x_119
        out_63 = out_62
        out_62 = x_119 = None
        input_31 = torch.nn.functional.relu(out_63, inplace=True)
        out_63 = None
        x_124 = torch.nn.functional.batch_norm(
            input_31,
            l_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_11_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_11_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_11_modules_identity_parameters_bias_
        ) = None
        x_125 = torch.conv2d(
            input_31,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_31 = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_125 = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_64 = 0 + x_126
        x_126 = None
        out_64 += x_124
        out_65 = out_64
        out_64 = x_124 = None
        input_32 = torch.nn.functional.relu(out_65, inplace=True)
        out_65 = None
        x_127 = torch.nn.functional.batch_norm(
            input_32,
            l_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_12_modules_identity_parameters_bias_
        ) = None
        x_128 = torch.conv2d(
            input_32,
            l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            768,
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_130 = torch.conv2d(
            input_32,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_32 = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_130 = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_129 += x_131
        out_66 = x_129
        x_129 = x_131 = None
        out_66 += x_127
        out_67 = out_66
        out_66 = x_127 = None
        input_33 = torch.nn.functional.relu(out_67, inplace=True)
        out_67 = None
        x_132 = torch.nn.functional.batch_norm(
            input_33,
            l_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_13_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_13_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_13_modules_identity_parameters_bias_
        ) = None
        x_133 = torch.conv2d(
            input_33,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_33 = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_134 = torch.nn.functional.batch_norm(
            x_133,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_133 = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_68 = 0 + x_134
        x_134 = None
        out_68 += x_132
        out_69 = out_68
        out_68 = x_132 = None
        input_34 = torch.nn.functional.relu(out_69, inplace=True)
        out_69 = None
        x_135 = torch.nn.functional.batch_norm(
            input_34,
            l_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_14_modules_identity_parameters_bias_
        ) = None
        x_136 = torch.conv2d(
            input_34,
            l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            768,
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_138 = torch.conv2d(
            input_34,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_34 = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_138 = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_137 += x_139
        out_70 = x_137
        x_137 = x_139 = None
        out_70 += x_135
        out_71 = out_70
        out_70 = x_135 = None
        input_35 = torch.nn.functional.relu(out_71, inplace=True)
        out_71 = None
        x_140 = torch.nn.functional.batch_norm(
            input_35,
            l_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_15_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_15_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_15_modules_identity_parameters_bias_
        ) = None
        x_141 = torch.conv2d(
            input_35,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_35 = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_141 = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_72 = 0 + x_142
        x_142 = None
        out_72 += x_140
        out_73 = out_72
        out_72 = x_140 = None
        input_36 = torch.nn.functional.relu(out_73, inplace=True)
        out_73 = None
        x_143 = torch.nn.functional.batch_norm(
            input_36,
            l_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_16_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_16_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_16_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_16_modules_identity_parameters_bias_
        ) = None
        x_144 = torch.conv2d(
            input_36,
            l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            768,
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_145 = torch.nn.functional.batch_norm(
            x_144,
            l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_144 = l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_146 = torch.conv2d(
            input_36,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_36 = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_147 = torch.nn.functional.batch_norm(
            x_146,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_146 = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_145 += x_147
        out_74 = x_145
        x_145 = x_147 = None
        out_74 += x_143
        out_75 = out_74
        out_74 = x_143 = None
        input_37 = torch.nn.functional.relu(out_75, inplace=True)
        out_75 = None
        x_148 = torch.nn.functional.batch_norm(
            input_37,
            l_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_17_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_17_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_17_modules_identity_parameters_bias_
        ) = None
        x_149 = torch.conv2d(
            input_37,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_37 = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_150 = torch.nn.functional.batch_norm(
            x_149,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_149 = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_76 = 0 + x_150
        x_150 = None
        out_76 += x_148
        out_77 = out_76
        out_76 = x_148 = None
        input_38 = torch.nn.functional.relu(out_77, inplace=True)
        out_77 = None
        x_151 = torch.nn.functional.batch_norm(
            input_38,
            l_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_18_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_18_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_18_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_18_modules_identity_parameters_bias_
        ) = None
        x_152 = torch.conv2d(
            input_38,
            l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            768,
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_152 = l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_154 = torch.conv2d(
            input_38,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_38 = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_153 += x_155
        out_78 = x_153
        x_153 = x_155 = None
        out_78 += x_151
        out_79 = out_78
        out_78 = x_151 = None
        input_39 = torch.nn.functional.relu(out_79, inplace=True)
        out_79 = None
        x_156 = torch.nn.functional.batch_norm(
            input_39,
            l_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_19_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_19_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_19_modules_identity_parameters_bias_
        ) = None
        x_157 = torch.conv2d(
            input_39,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_39 = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_158 = torch.nn.functional.batch_norm(
            x_157,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_157 = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_80 = 0 + x_158
        x_158 = None
        out_80 += x_156
        out_81 = out_80
        out_80 = x_156 = None
        input_40 = torch.nn.functional.relu(out_81, inplace=True)
        out_81 = None
        x_159 = torch.conv2d(
            input_40,
            l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            768,
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_160 = torch.nn.functional.batch_norm(
            x_159,
            l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_159 = l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_161 = torch.conv2d(
            input_40,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            768,
        )
        input_40 = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_162 = torch.nn.functional.batch_norm(
            x_161,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_161 = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_160 += x_162
        out_82 = x_160
        x_160 = x_162 = None
        out_82 += 0
        out_83 = out_82
        out_82 = None
        input_41 = torch.nn.functional.relu(out_83, inplace=True)
        out_83 = None
        x_163 = torch.conv2d(
            input_41,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_41 = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (None)
        x_164 = torch.nn.functional.batch_norm(
            x_163,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_163 = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_84 = 0 + x_164
        x_164 = None
        out_84 += 0
        out_85 = out_84
        out_84 = None
        input_42 = torch.nn.functional.relu(out_85, inplace=True)
        out_85 = None
        x_165 = torch.nn.functional.adaptive_avg_pool2d(input_42, 1)
        input_42 = None
        x_166 = x_165.flatten(1, -1)
        x_165 = None
        x_167 = torch.nn.functional.dropout(x_166, 0.0, False, False)
        x_166 = None
        x_168 = torch._C._nn.linear(
            x_167,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_167 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_168,)
