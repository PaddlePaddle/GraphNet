import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_conv_1x1_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv_1x1_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_conv_1x1_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_conv_1x1_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_conv_1x1_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_conv_1x1_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_conv_kxk_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv_kxk_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_conv_kxk_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_conv_kxk_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_conv_kxk_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_conv_kxk_modules_bn_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_identity_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_identity_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_1_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_0_modules_2_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_0_modules_2_modules_identity_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_identity_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_2_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_0_modules_3_modules_identity_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_identity_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_3_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_1_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_2_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_2_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_2_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_3_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_3_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_4_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_4_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_4_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_1_modules_5_modules_identity_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_identity_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_5_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_1_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_2_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_2_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_2_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_3_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_3_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_4_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_4_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_4_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_5_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_5_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_6_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_6_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_6_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_7_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_7_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_8_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_8_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_8_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_9_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_9_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_10_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_10_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_10_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_11_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_11_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_12_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_12_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_12_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_13_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_13_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_14_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_14_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_14_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_2_modules_15_modules_identity_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_identity_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_15_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_bias_
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stem_modules_conv_1x1_modules_conv_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_conv_1x1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_conv_1x1_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_conv_1x1_modules_bn_parameters_bias_ = None
        x_2 = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_stem_modules_conv_kxk_modules_conv_parameters_weight_
        ) = None
        x_3 = torch.nn.functional.batch_norm(
            x_2,
            l_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_2 = (
            l_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_conv_kxk_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_conv_kxk_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_conv_kxk_modules_bn_parameters_bias_ = None
        x_4 = x_1 + x_3
        x_1 = x_3 = None
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_8 = torch.conv2d(
            x_5,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_9 = torch.nn.functional.batch_norm(
            x_8,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_8 = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_10 = x_7 + x_9
        x_7 = x_9 = None
        input_1 = torch.nn.functional.relu(x_10, inplace=True)
        x_10 = None
        x_11 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_0_modules_1_modules_identity_parameters_bias_
        ) = None
        x_12 = torch.conv2d(
            input_1,
            l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_12 = l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_14 = torch.conv2d(
            input_1,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        input_1 = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_14 = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_16 = x_13 + x_15
        x_13 = x_15 = None
        x_16 += x_11
        x_17 = x_16
        x_16 = x_11 = None
        input_2 = torch.nn.functional.relu(x_17, inplace=True)
        x_17 = None
        x_18 = torch.nn.functional.batch_norm(
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
        x_19 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_2_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_21 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_2 = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_21 = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_23 = x_20 + x_22
        x_20 = x_22 = None
        x_23 += x_18
        x_24 = x_23
        x_23 = x_18 = None
        input_3 = torch.nn.functional.relu(x_24, inplace=True)
        x_24 = None
        x_25 = torch.nn.functional.batch_norm(
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
        x_26 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_3_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_28 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        input_3 = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_30 = x_27 + x_29
        x_27 = x_29 = None
        x_30 += x_25
        x_31 = x_30
        x_30 = x_25 = None
        input_4 = torch.nn.functional.relu(x_31, inplace=True)
        x_31 = None
        x_32 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_34 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_4 = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_36 = x_33 + x_35
        x_33 = x_35 = None
        input_5 = torch.nn.functional.relu(x_36, inplace=True)
        x_36 = None
        x_37 = torch.nn.functional.batch_norm(
            input_5,
            l_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_1_modules_identity_parameters_bias_
        ) = None
        x_38 = torch.conv2d(
            input_5,
            l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_40 = torch.conv2d(
            input_5,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        input_5 = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_42 = x_39 + x_41
        x_39 = x_41 = None
        x_42 += x_37
        x_43 = x_42
        x_42 = x_37 = None
        input_6 = torch.nn.functional.relu(x_43, inplace=True)
        x_43 = None
        x_44 = torch.nn.functional.batch_norm(
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
        x_45 = torch.conv2d(
            input_6,
            l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_47 = torch.conv2d(
            input_6,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_6 = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_49 = x_46 + x_48
        x_46 = x_48 = None
        x_49 += x_44
        x_50 = x_49
        x_49 = x_44 = None
        input_7 = torch.nn.functional.relu(x_50, inplace=True)
        x_50 = None
        x_51 = torch.nn.functional.batch_norm(
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
        x_52 = torch.conv2d(
            input_7,
            l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_3_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_54 = torch.conv2d(
            input_7,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        input_7 = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_56 = x_53 + x_55
        x_53 = x_55 = None
        x_56 += x_51
        x_57 = x_56
        x_56 = x_51 = None
        input_8 = torch.nn.functional.relu(x_57, inplace=True)
        x_57 = None
        x_58 = torch.nn.functional.batch_norm(
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
        x_59 = torch.conv2d(
            input_8,
            l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_59 = l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_4_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_61 = torch.conv2d(
            input_8,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_8 = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_63 = x_60 + x_62
        x_60 = x_62 = None
        x_63 += x_58
        x_64 = x_63
        x_63 = x_58 = None
        input_9 = torch.nn.functional.relu(x_64, inplace=True)
        x_64 = None
        x_65 = torch.nn.functional.batch_norm(
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
        x_66 = torch.conv2d(
            input_9,
            l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_5_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_68 = torch.conv2d(
            input_9,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        input_9 = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_70 = x_67 + x_69
        x_67 = x_69 = None
        x_70 += x_65
        x_71 = x_70
        x_70 = x_65 = None
        input_10 = torch.nn.functional.relu(x_71, inplace=True)
        x_71 = None
        x_72 = torch.conv2d(
            input_10,
            l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_74 = torch.conv2d(
            input_10,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_10 = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_75 = torch.nn.functional.batch_norm(
            x_74,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_74 = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_76 = x_73 + x_75
        x_73 = x_75 = None
        input_11 = torch.nn.functional.relu(x_76, inplace=True)
        x_76 = None
        x_77 = torch.nn.functional.batch_norm(
            input_11,
            l_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_identity_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_identity_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_identity_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_identity_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_1_modules_identity_parameters_bias_
        ) = None
        x_78 = torch.conv2d(
            input_11,
            l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_79 = torch.nn.functional.batch_norm(
            x_78,
            l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_78 = l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_80 = torch.conv2d(
            input_11,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        input_11 = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_82 = x_79 + x_81
        x_79 = x_81 = None
        x_82 += x_77
        x_83 = x_82
        x_82 = x_77 = None
        input_12 = torch.nn.functional.relu(x_83, inplace=True)
        x_83 = None
        x_84 = torch.nn.functional.batch_norm(
            input_12,
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
        x_85 = torch.conv2d(
            input_12,
            l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_85 = l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_87 = torch.conv2d(
            input_12,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_12 = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_88 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_87 = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_89 = x_86 + x_88
        x_86 = x_88 = None
        x_89 += x_84
        x_90 = x_89
        x_89 = x_84 = None
        input_13 = torch.nn.functional.relu(x_90, inplace=True)
        x_90 = None
        x_91 = torch.nn.functional.batch_norm(
            input_13,
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
        x_92 = torch.conv2d(
            input_13,
            l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_94 = torch.conv2d(
            input_13,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        input_13 = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_96 = x_93 + x_95
        x_93 = x_95 = None
        x_96 += x_91
        x_97 = x_96
        x_96 = x_91 = None
        input_14 = torch.nn.functional.relu(x_97, inplace=True)
        x_97 = None
        x_98 = torch.nn.functional.batch_norm(
            input_14,
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
        x_99 = torch.conv2d(
            input_14,
            l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_99 = l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_101 = torch.conv2d(
            input_14,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_14 = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_103 = x_100 + x_102
        x_100 = x_102 = None
        x_103 += x_98
        x_104 = x_103
        x_103 = x_98 = None
        input_15 = torch.nn.functional.relu(x_104, inplace=True)
        x_104 = None
        x_105 = torch.nn.functional.batch_norm(
            input_15,
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
        x_106 = torch.conv2d(
            input_15,
            l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_107 = torch.nn.functional.batch_norm(
            x_106,
            l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_106 = l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_108 = torch.conv2d(
            input_15,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        input_15 = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_108 = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_110 = x_107 + x_109
        x_107 = x_109 = None
        x_110 += x_105
        x_111 = x_110
        x_110 = x_105 = None
        input_16 = torch.nn.functional.relu(x_111, inplace=True)
        x_111 = None
        x_112 = torch.nn.functional.batch_norm(
            input_16,
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
        x_113 = torch.conv2d(
            input_16,
            l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_115 = torch.conv2d(
            input_16,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_16 = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_115 = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_117 = x_114 + x_116
        x_114 = x_116 = None
        x_117 += x_112
        x_118 = x_117
        x_117 = x_112 = None
        input_17 = torch.nn.functional.relu(x_118, inplace=True)
        x_118 = None
        x_119 = torch.nn.functional.batch_norm(
            input_17,
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
        x_120 = torch.conv2d(
            input_17,
            l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_122 = torch.conv2d(
            input_17,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        input_17 = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_123 = torch.nn.functional.batch_norm(
            x_122,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_122 = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_124 = x_121 + x_123
        x_121 = x_123 = None
        x_124 += x_119
        x_125 = x_124
        x_124 = x_119 = None
        input_18 = torch.nn.functional.relu(x_125, inplace=True)
        x_125 = None
        x_126 = torch.nn.functional.batch_norm(
            input_18,
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
        x_127 = torch.conv2d(
            input_18,
            l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_127 = l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_129 = torch.conv2d(
            input_18,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_18 = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_130 = torch.nn.functional.batch_norm(
            x_129,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_129 = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_131 = x_128 + x_130
        x_128 = x_130 = None
        x_131 += x_126
        x_132 = x_131
        x_131 = x_126 = None
        input_19 = torch.nn.functional.relu(x_132, inplace=True)
        x_132 = None
        x_133 = torch.nn.functional.batch_norm(
            input_19,
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
        x_134 = torch.conv2d(
            input_19,
            l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_136 = torch.conv2d(
            input_19,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        input_19 = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_138 = x_135 + x_137
        x_135 = x_137 = None
        x_138 += x_133
        x_139 = x_138
        x_138 = x_133 = None
        input_20 = torch.nn.functional.relu(x_139, inplace=True)
        x_139 = None
        x_140 = torch.nn.functional.batch_norm(
            input_20,
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
        x_141 = torch.conv2d(
            input_20,
            l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_141 = l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_143 = torch.conv2d(
            input_20,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_20 = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_144 = torch.nn.functional.batch_norm(
            x_143,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_143 = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_145 = x_142 + x_144
        x_142 = x_144 = None
        x_145 += x_140
        x_146 = x_145
        x_145 = x_140 = None
        input_21 = torch.nn.functional.relu(x_146, inplace=True)
        x_146 = None
        x_147 = torch.nn.functional.batch_norm(
            input_21,
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
        x_148 = torch.conv2d(
            input_21,
            l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_150 = torch.conv2d(
            input_21,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        input_21 = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_151 = torch.nn.functional.batch_norm(
            x_150,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_150 = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_152 = x_149 + x_151
        x_149 = x_151 = None
        x_152 += x_147
        x_153 = x_152
        x_152 = x_147 = None
        input_22 = torch.nn.functional.relu(x_153, inplace=True)
        x_153 = None
        x_154 = torch.nn.functional.batch_norm(
            input_22,
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
        x_155 = torch.conv2d(
            input_22,
            l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_156 = torch.nn.functional.batch_norm(
            x_155,
            l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_155 = l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_157 = torch.conv2d(
            input_22,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_22 = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_158 = torch.nn.functional.batch_norm(
            x_157,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_157 = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_159 = x_156 + x_158
        x_156 = x_158 = None
        x_159 += x_154
        x_160 = x_159
        x_159 = x_154 = None
        input_23 = torch.nn.functional.relu(x_160, inplace=True)
        x_160 = None
        x_161 = torch.nn.functional.batch_norm(
            input_23,
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
        x_162 = torch.conv2d(
            input_23,
            l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_163 = torch.nn.functional.batch_norm(
            x_162,
            l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_162 = l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_164 = torch.conv2d(
            input_23,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        input_23 = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_165 = torch.nn.functional.batch_norm(
            x_164,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_164 = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_166 = x_163 + x_165
        x_163 = x_165 = None
        x_166 += x_161
        x_167 = x_166
        x_166 = x_161 = None
        input_24 = torch.nn.functional.relu(x_167, inplace=True)
        x_167 = None
        x_168 = torch.nn.functional.batch_norm(
            input_24,
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
        x_169 = torch.conv2d(
            input_24,
            l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_170 = torch.nn.functional.batch_norm(
            x_169,
            l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_169 = l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_171 = torch.conv2d(
            input_24,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_24 = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_172 = torch.nn.functional.batch_norm(
            x_171,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_171 = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_173 = x_170 + x_172
        x_170 = x_172 = None
        x_173 += x_168
        x_174 = x_173
        x_173 = x_168 = None
        input_25 = torch.nn.functional.relu(x_174, inplace=True)
        x_174 = None
        x_175 = torch.nn.functional.batch_norm(
            input_25,
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
        x_176 = torch.conv2d(
            input_25,
            l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            4,
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_177 = torch.nn.functional.batch_norm(
            x_176,
            l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_176 = l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_178 = torch.conv2d(
            input_25,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        input_25 = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_179 = torch.nn.functional.batch_norm(
            x_178,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_178 = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_180 = x_177 + x_179
        x_177 = x_179 = None
        x_180 += x_175
        x_181 = x_180
        x_180 = x_175 = None
        input_26 = torch.nn.functional.relu(x_181, inplace=True)
        x_181 = None
        x_182 = torch.conv2d(
            input_26,
            l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_conv_parameters_weight_ = (
            None
        )
        x_183 = torch.nn.functional.batch_norm(
            x_182,
            l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_182 = l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv_1x1_modules_bn_parameters_bias_ = (None)
        x_184 = torch.conv2d(
            input_26,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_26 = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_conv_parameters_weight_ = (None)
        x_185 = torch.nn.functional.batch_norm(
            x_184,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_184 = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_bn_parameters_bias_ = (None)
        x_186 = x_183 + x_185
        x_183 = x_185 = None
        input_27 = torch.nn.functional.relu(x_186, inplace=True)
        x_186 = None
        x_187 = torch.nn.functional.adaptive_avg_pool2d(input_27, 1)
        input_27 = None
        x_188 = x_187.flatten(1, -1)
        x_187 = None
        x_189 = torch.nn.functional.dropout(x_188, 0.0, False, False)
        x_188 = None
        x_190 = torch._C._nn.linear(
            x_189,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_189 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_190,)
