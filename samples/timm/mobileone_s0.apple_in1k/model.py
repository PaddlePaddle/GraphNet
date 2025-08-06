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
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_15_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_15_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_identity_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_identity_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_identity_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_identity_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_identity_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_identity_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_identity_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_1_modules_identity_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_
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
            48,
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
            48,
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
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
        x_9 = torch.conv2d(
            x_4,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            48,
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_2 += x_10
        out_3 = out_2
        out_2 = x_10 = None
        x_11 = torch.conv2d(
            x_4,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            48,
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_12 = torch.nn.functional.batch_norm(
            x_11,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_11 = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_3 += x_12
        out_4 = out_3
        out_3 = x_12 = None
        x_13 = torch.conv2d(
            x_4,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            48,
        )
        x_4 = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_14 = torch.nn.functional.batch_norm(
            x_13,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_13 = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_4 += x_14
        out_5 = out_4
        out_4 = x_14 = None
        out_5 += 0
        out_6 = out_5
        out_5 = None
        input_1 = torch.nn.functional.relu(out_6, inplace=True)
        out_6 = None
        x_15 = torch.nn.functional.batch_norm(
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
        x_16 = torch.conv2d(
            input_1,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_7 = 0 + x_17
        x_17 = None
        x_18 = torch.conv2d(
            input_1,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_7 += x_19
        out_8 = out_7
        out_7 = x_19 = None
        x_20 = torch.conv2d(
            input_1,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_8 += x_21
        out_9 = out_8
        out_8 = x_21 = None
        x_22 = torch.conv2d(
            input_1,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_1 = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_9 += x_23
        out_10 = out_9
        out_9 = x_23 = None
        out_10 += x_15
        out_11 = out_10
        out_10 = x_15 = None
        input_2 = torch.nn.functional.relu(out_11, inplace=True)
        out_11 = None
        x_24 = torch.nn.functional.batch_norm(
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
        x_25 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            48,
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_2_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_27 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_28 = torch.nn.functional.batch_norm(
            x_27,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_27 = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_26 += x_28
        out_12 = x_26
        x_26 = x_28 = None
        x_29 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_12 += x_30
        out_13 = out_12
        out_12 = x_30 = None
        x_31 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_13 += x_32
        out_14 = out_13
        out_13 = x_32 = None
        x_33 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        input_2 = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_14 += x_34
        out_15 = out_14
        out_14 = x_34 = None
        out_15 += x_24
        out_16 = out_15
        out_15 = x_24 = None
        input_3 = torch.nn.functional.relu(out_16, inplace=True)
        out_16 = None
        x_35 = torch.nn.functional.batch_norm(
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
        x_36 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_17 = 0 + x_37
        x_37 = None
        x_38 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_17 += x_39
        out_18 = out_17
        out_17 = x_39 = None
        x_40 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_18 += x_41
        out_19 = out_18
        out_18 = x_41 = None
        x_42 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_3 = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_19 += x_43
        out_20 = out_19
        out_19 = x_43 = None
        out_20 += x_35
        out_21 = out_20
        out_20 = x_35 = None
        input_4 = torch.nn.functional.relu(out_21, inplace=True)
        out_21 = None
        x_44 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            48,
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_45 = torch.nn.functional.batch_norm(
            x_44,
            l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_44 = l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_46 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            48,
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_46 = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_45 += x_47
        out_22 = x_45
        x_45 = x_47 = None
        x_48 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            48,
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_48 = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_22 += x_49
        out_23 = out_22
        out_22 = x_49 = None
        x_50 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            48,
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_23 += x_51
        out_24 = out_23
        out_23 = x_51 = None
        x_52 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            48,
        )
        input_4 = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_24 += x_53
        out_25 = out_24
        out_24 = x_53 = None
        out_25 += 0
        out_26 = out_25
        out_25 = None
        input_5 = torch.nn.functional.relu(out_26, inplace=True)
        out_26 = None
        x_54 = torch.conv2d(
            input_5,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_27 = 0 + x_55
        x_55 = None
        x_56 = torch.conv2d(
            input_5,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_27 += x_57
        out_28 = out_27
        out_27 = x_57 = None
        x_58 = torch.conv2d(
            input_5,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_28 += x_59
        out_29 = out_28
        out_28 = x_59 = None
        x_60 = torch.conv2d(
            input_5,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_5 = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_60 = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_29 += x_61
        out_30 = out_29
        out_29 = x_61 = None
        out_30 += 0
        out_31 = out_30
        out_30 = None
        input_6 = torch.nn.functional.relu(out_31, inplace=True)
        out_31 = None
        x_62 = torch.nn.functional.batch_norm(
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
        x_63 = torch.conv2d(
            input_6,
            l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_65 = torch.conv2d(
            input_6,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_65 = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_64 += x_66
        out_32 = x_64
        x_64 = x_66 = None
        x_67 = torch.conv2d(
            input_6,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_68 = torch.nn.functional.batch_norm(
            x_67,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_67 = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_32 += x_68
        out_33 = out_32
        out_32 = x_68 = None
        x_69 = torch.conv2d(
            input_6,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_33 += x_70
        out_34 = out_33
        out_33 = x_70 = None
        x_71 = torch.conv2d(
            input_6,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        input_6 = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_72 = torch.nn.functional.batch_norm(
            x_71,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_71 = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_34 += x_72
        out_35 = out_34
        out_34 = x_72 = None
        out_35 += x_62
        out_36 = out_35
        out_35 = x_62 = None
        input_7 = torch.nn.functional.relu(out_36, inplace=True)
        out_36 = None
        x_73 = torch.nn.functional.batch_norm(
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
        x_74 = torch.conv2d(
            input_7,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_75 = torch.nn.functional.batch_norm(
            x_74,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_74 = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_37 = 0 + x_75
        x_75 = None
        x_76 = torch.conv2d(
            input_7,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_37 += x_77
        out_38 = out_37
        out_37 = x_77 = None
        x_78 = torch.conv2d(
            input_7,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_79 = torch.nn.functional.batch_norm(
            x_78,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_78 = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_38 += x_79
        out_39 = out_38
        out_38 = x_79 = None
        x_80 = torch.conv2d(
            input_7,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_7 = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_39 += x_81
        out_40 = out_39
        out_39 = x_81 = None
        out_40 += x_73
        out_41 = out_40
        out_40 = x_73 = None
        input_8 = torch.nn.functional.relu(out_41, inplace=True)
        out_41 = None
        x_82 = torch.nn.functional.batch_norm(
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
        x_83 = torch.conv2d(
            input_8,
            l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_4_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_85 = torch.conv2d(
            input_8,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_85 = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_84 += x_86
        out_42 = x_84
        x_84 = x_86 = None
        x_87 = torch.conv2d(
            input_8,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_88 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_87 = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_42 += x_88
        out_43 = out_42
        out_42 = x_88 = None
        x_89 = torch.conv2d(
            input_8,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_89 = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_43 += x_90
        out_44 = out_43
        out_43 = x_90 = None
        x_91 = torch.conv2d(
            input_8,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        input_8 = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_44 += x_92
        out_45 = out_44
        out_44 = x_92 = None
        out_45 += x_82
        out_46 = out_45
        out_45 = x_82 = None
        input_9 = torch.nn.functional.relu(out_46, inplace=True)
        out_46 = None
        x_93 = torch.nn.functional.batch_norm(
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
        x_94 = torch.conv2d(
            input_9,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_47 = 0 + x_95
        x_95 = None
        x_96 = torch.conv2d(
            input_9,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_96 = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_47 += x_97
        out_48 = out_47
        out_47 = x_97 = None
        x_98 = torch.conv2d(
            input_9,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_98 = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_48 += x_99
        out_49 = out_48
        out_48 = x_99 = None
        x_100 = torch.conv2d(
            input_9,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_9 = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_100 = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_49 += x_101
        out_50 = out_49
        out_49 = x_101 = None
        out_50 += x_93
        out_51 = out_50
        out_50 = x_93 = None
        input_10 = torch.nn.functional.relu(out_51, inplace=True)
        out_51 = None
        x_102 = torch.nn.functional.batch_norm(
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
        x_103 = torch.conv2d(
            input_10,
            l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_104 = torch.nn.functional.batch_norm(
            x_103,
            l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_103 = l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_6_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_105 = torch.conv2d(
            input_10,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_106 = torch.nn.functional.batch_norm(
            x_105,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_105 = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_104 += x_106
        out_52 = x_104
        x_104 = x_106 = None
        x_107 = torch.conv2d(
            input_10,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_52 += x_108
        out_53 = out_52
        out_52 = x_108 = None
        x_109 = torch.conv2d(
            input_10,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_110 = torch.nn.functional.batch_norm(
            x_109,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_109 = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_53 += x_110
        out_54 = out_53
        out_53 = x_110 = None
        x_111 = torch.conv2d(
            input_10,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        input_10 = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_112 = torch.nn.functional.batch_norm(
            x_111,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_111 = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_54 += x_112
        out_55 = out_54
        out_54 = x_112 = None
        out_55 += x_102
        out_56 = out_55
        out_55 = x_102 = None
        input_11 = torch.nn.functional.relu(out_56, inplace=True)
        out_56 = None
        x_113 = torch.nn.functional.batch_norm(
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
        x_114 = torch.conv2d(
            input_11,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_115 = torch.nn.functional.batch_norm(
            x_114,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_114 = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_57 = 0 + x_115
        x_115 = None
        x_116 = torch.conv2d(
            input_11,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_117 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_116 = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_57 += x_117
        out_58 = out_57
        out_57 = x_117 = None
        x_118 = torch.conv2d(
            input_11,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_119 = torch.nn.functional.batch_norm(
            x_118,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_118 = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_58 += x_119
        out_59 = out_58
        out_58 = x_119 = None
        x_120 = torch.conv2d(
            input_11,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_11 = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_59 += x_121
        out_60 = out_59
        out_59 = x_121 = None
        out_60 += x_113
        out_61 = out_60
        out_60 = x_113 = None
        input_12 = torch.nn.functional.relu(out_61, inplace=True)
        out_61 = None
        x_122 = torch.nn.functional.batch_norm(
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
        x_123 = torch.conv2d(
            input_12,
            l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_8_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_125 = torch.conv2d(
            input_12,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_125 = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_124 += x_126
        out_62 = x_124
        x_124 = x_126 = None
        x_127 = torch.conv2d(
            input_12,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_127 = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_62 += x_128
        out_63 = out_62
        out_62 = x_128 = None
        x_129 = torch.conv2d(
            input_12,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_130 = torch.nn.functional.batch_norm(
            x_129,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_129 = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_63 += x_130
        out_64 = out_63
        out_63 = x_130 = None
        x_131 = torch.conv2d(
            input_12,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        input_12 = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_64 += x_132
        out_65 = out_64
        out_64 = x_132 = None
        out_65 += x_122
        out_66 = out_65
        out_65 = x_122 = None
        input_13 = torch.nn.functional.relu(out_66, inplace=True)
        out_66 = None
        x_133 = torch.nn.functional.batch_norm(
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
        x_134 = torch.conv2d(
            input_13,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_67 = 0 + x_135
        x_135 = None
        x_136 = torch.conv2d(
            input_13,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_67 += x_137
        out_68 = out_67
        out_67 = x_137 = None
        x_138 = torch.conv2d(
            input_13,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_138 = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_68 += x_139
        out_69 = out_68
        out_68 = x_139 = None
        x_140 = torch.conv2d(
            input_13,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_13 = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_141 = torch.nn.functional.batch_norm(
            x_140,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_140 = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_69 += x_141
        out_70 = out_69
        out_69 = x_141 = None
        out_70 += x_133
        out_71 = out_70
        out_70 = x_133 = None
        input_14 = torch.nn.functional.relu(out_71, inplace=True)
        out_71 = None
        x_142 = torch.nn.functional.batch_norm(
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
        x_143 = torch.conv2d(
            input_14,
            l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_144 = torch.nn.functional.batch_norm(
            x_143,
            l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_143 = l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_10_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_145 = torch.conv2d(
            input_14,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_146 = torch.nn.functional.batch_norm(
            x_145,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_145 = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_144 += x_146
        out_72 = x_144
        x_144 = x_146 = None
        x_147 = torch.conv2d(
            input_14,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_148 = torch.nn.functional.batch_norm(
            x_147,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_147 = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_72 += x_148
        out_73 = out_72
        out_72 = x_148 = None
        x_149 = torch.conv2d(
            input_14,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_150 = torch.nn.functional.batch_norm(
            x_149,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_149 = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_73 += x_150
        out_74 = out_73
        out_73 = x_150 = None
        x_151 = torch.conv2d(
            input_14,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        input_14 = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_151 = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_74 += x_152
        out_75 = out_74
        out_74 = x_152 = None
        out_75 += x_142
        out_76 = out_75
        out_75 = x_142 = None
        input_15 = torch.nn.functional.relu(out_76, inplace=True)
        out_76 = None
        x_153 = torch.nn.functional.batch_norm(
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
        x_154 = torch.conv2d(
            input_15,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_77 = 0 + x_155
        x_155 = None
        x_156 = torch.conv2d(
            input_15,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_157 = torch.nn.functional.batch_norm(
            x_156,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_156 = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_77 += x_157
        out_78 = out_77
        out_77 = x_157 = None
        x_158 = torch.conv2d(
            input_15,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_159 = torch.nn.functional.batch_norm(
            x_158,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_158 = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_78 += x_159
        out_79 = out_78
        out_78 = x_159 = None
        x_160 = torch.conv2d(
            input_15,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_15 = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_161 = torch.nn.functional.batch_norm(
            x_160,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_160 = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_79 += x_161
        out_80 = out_79
        out_79 = x_161 = None
        out_80 += x_153
        out_81 = out_80
        out_80 = x_153 = None
        input_16 = torch.nn.functional.relu(out_81, inplace=True)
        out_81 = None
        x_162 = torch.nn.functional.batch_norm(
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
        x_163 = torch.conv2d(
            input_16,
            l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_164 = torch.nn.functional.batch_norm(
            x_163,
            l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_163 = l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_12_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_165 = torch.conv2d(
            input_16,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_166 = torch.nn.functional.batch_norm(
            x_165,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_165 = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_164 += x_166
        out_82 = x_164
        x_164 = x_166 = None
        x_167 = torch.conv2d(
            input_16,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_168 = torch.nn.functional.batch_norm(
            x_167,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_167 = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_82 += x_168
        out_83 = out_82
        out_82 = x_168 = None
        x_169 = torch.conv2d(
            input_16,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_170 = torch.nn.functional.batch_norm(
            x_169,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_169 = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_83 += x_170
        out_84 = out_83
        out_83 = x_170 = None
        x_171 = torch.conv2d(
            input_16,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        input_16 = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_172 = torch.nn.functional.batch_norm(
            x_171,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_171 = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_84 += x_172
        out_85 = out_84
        out_84 = x_172 = None
        out_85 += x_162
        out_86 = out_85
        out_85 = x_162 = None
        input_17 = torch.nn.functional.relu(out_86, inplace=True)
        out_86 = None
        x_173 = torch.nn.functional.batch_norm(
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
        x_174 = torch.conv2d(
            input_17,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_175 = torch.nn.functional.batch_norm(
            x_174,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_174 = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_87 = 0 + x_175
        x_175 = None
        x_176 = torch.conv2d(
            input_17,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_177 = torch.nn.functional.batch_norm(
            x_176,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_176 = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_87 += x_177
        out_88 = out_87
        out_87 = x_177 = None
        x_178 = torch.conv2d(
            input_17,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_179 = torch.nn.functional.batch_norm(
            x_178,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_178 = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_88 += x_179
        out_89 = out_88
        out_88 = x_179 = None
        x_180 = torch.conv2d(
            input_17,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_17 = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_181 = torch.nn.functional.batch_norm(
            x_180,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_180 = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_89 += x_181
        out_90 = out_89
        out_89 = x_181 = None
        out_90 += x_173
        out_91 = out_90
        out_90 = x_173 = None
        input_18 = torch.nn.functional.relu(out_91, inplace=True)
        out_91 = None
        x_182 = torch.nn.functional.batch_norm(
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
        x_183 = torch.conv2d(
            input_18,
            l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_184 = torch.nn.functional.batch_norm(
            x_183,
            l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_183 = l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_14_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_185 = torch.conv2d(
            input_18,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_186 = torch.nn.functional.batch_norm(
            x_185,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_185 = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_184 += x_186
        out_92 = x_184
        x_184 = x_186 = None
        x_187 = torch.conv2d(
            input_18,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_188 = torch.nn.functional.batch_norm(
            x_187,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_187 = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_92 += x_188
        out_93 = out_92
        out_92 = x_188 = None
        x_189 = torch.conv2d(
            input_18,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_190 = torch.nn.functional.batch_norm(
            x_189,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_189 = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_93 += x_190
        out_94 = out_93
        out_93 = x_190 = None
        x_191 = torch.conv2d(
            input_18,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        input_18 = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_192 = torch.nn.functional.batch_norm(
            x_191,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_191 = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_94 += x_192
        out_95 = out_94
        out_94 = x_192 = None
        out_95 += x_182
        out_96 = out_95
        out_95 = x_182 = None
        input_19 = torch.nn.functional.relu(out_96, inplace=True)
        out_96 = None
        x_193 = torch.nn.functional.batch_norm(
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
        x_194 = torch.conv2d(
            input_19,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_195 = torch.nn.functional.batch_norm(
            x_194,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_194 = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_97 = 0 + x_195
        x_195 = None
        x_196 = torch.conv2d(
            input_19,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_197 = torch.nn.functional.batch_norm(
            x_196,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_196 = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_97 += x_197
        out_98 = out_97
        out_97 = x_197 = None
        x_198 = torch.conv2d(
            input_19,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_199 = torch.nn.functional.batch_norm(
            x_198,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_198 = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_98 += x_199
        out_99 = out_98
        out_98 = x_199 = None
        x_200 = torch.conv2d(
            input_19,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_19 = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_201 = torch.nn.functional.batch_norm(
            x_200,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_200 = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_99 += x_201
        out_100 = out_99
        out_99 = x_201 = None
        out_100 += x_193
        out_101 = out_100
        out_100 = x_193 = None
        input_20 = torch.nn.functional.relu(out_101, inplace=True)
        out_101 = None
        x_202 = torch.conv2d(
            input_20,
            l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_203 = torch.nn.functional.batch_norm(
            x_202,
            l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_202 = l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_204 = torch.conv2d(
            input_20,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_205 = torch.nn.functional.batch_norm(
            x_204,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_204 = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_203 += x_205
        out_102 = x_203
        x_203 = x_205 = None
        x_206 = torch.conv2d(
            input_20,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_207 = torch.nn.functional.batch_norm(
            x_206,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_206 = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_102 += x_207
        out_103 = out_102
        out_102 = x_207 = None
        x_208 = torch.conv2d(
            input_20,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_209 = torch.nn.functional.batch_norm(
            x_208,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_208 = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_103 += x_209
        out_104 = out_103
        out_103 = x_209 = None
        x_210 = torch.conv2d(
            input_20,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            128,
        )
        input_20 = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_211 = torch.nn.functional.batch_norm(
            x_210,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_210 = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_104 += x_211
        out_105 = out_104
        out_104 = x_211 = None
        out_105 += 0
        out_106 = out_105
        out_105 = None
        input_21 = torch.nn.functional.relu(out_106, inplace=True)
        out_106 = None
        x_212 = torch.conv2d(
            input_21,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_213 = torch.nn.functional.batch_norm(
            x_212,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_212 = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_107 = 0 + x_213
        x_213 = None
        x_214 = torch.conv2d(
            input_21,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_215 = torch.nn.functional.batch_norm(
            x_214,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_214 = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_107 += x_215
        out_108 = out_107
        out_107 = x_215 = None
        x_216 = torch.conv2d(
            input_21,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_217 = torch.nn.functional.batch_norm(
            x_216,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_216 = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_108 += x_217
        out_109 = out_108
        out_108 = x_217 = None
        x_218 = torch.conv2d(
            input_21,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_21 = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_219 = torch.nn.functional.batch_norm(
            x_218,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_218 = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_109 += x_219
        out_110 = out_109
        out_109 = x_219 = None
        out_110 += 0
        out_111 = out_110
        out_110 = None
        input_22 = torch.nn.functional.relu(out_111, inplace=True)
        out_111 = None
        x_220 = torch.nn.functional.batch_norm(
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
        x_221 = torch.conv2d(
            input_22,
            l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_222 = torch.nn.functional.batch_norm(
            x_221,
            l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_221 = l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_223 = torch.conv2d(
            input_22,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_224 = torch.nn.functional.batch_norm(
            x_223,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_223 = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_222 += x_224
        out_112 = x_222
        x_222 = x_224 = None
        x_225 = torch.conv2d(
            input_22,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_226 = torch.nn.functional.batch_norm(
            x_225,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_225 = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_112 += x_226
        out_113 = out_112
        out_112 = x_226 = None
        x_227 = torch.conv2d(
            input_22,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_228 = torch.nn.functional.batch_norm(
            x_227,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_227 = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_113 += x_228
        out_114 = out_113
        out_113 = x_228 = None
        x_229 = torch.conv2d(
            input_22,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        input_22 = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_230 = torch.nn.functional.batch_norm(
            x_229,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_229 = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_114 += x_230
        out_115 = out_114
        out_114 = x_230 = None
        out_115 += x_220
        out_116 = out_115
        out_115 = x_220 = None
        input_23 = torch.nn.functional.relu(out_116, inplace=True)
        out_116 = None
        x_231 = torch.nn.functional.batch_norm(
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
        x_232 = torch.conv2d(
            input_23,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_233 = torch.nn.functional.batch_norm(
            x_232,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_232 = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_117 = 0 + x_233
        x_233 = None
        x_234 = torch.conv2d(
            input_23,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_235 = torch.nn.functional.batch_norm(
            x_234,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_234 = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_117 += x_235
        out_118 = out_117
        out_117 = x_235 = None
        x_236 = torch.conv2d(
            input_23,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_237 = torch.nn.functional.batch_norm(
            x_236,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_236 = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_118 += x_237
        out_119 = out_118
        out_118 = x_237 = None
        x_238 = torch.conv2d(
            input_23,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_23 = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_239 = torch.nn.functional.batch_norm(
            x_238,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_238 = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_119 += x_239
        out_120 = out_119
        out_119 = x_239 = None
        out_120 += x_231
        out_121 = out_120
        out_120 = x_231 = None
        input_24 = torch.nn.functional.relu(out_121, inplace=True)
        out_121 = None
        x_240 = torch.nn.functional.batch_norm(
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
        x_241 = torch.conv2d(
            input_24,
            l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_242 = torch.nn.functional.batch_norm(
            x_241,
            l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_241 = l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_243 = torch.conv2d(
            input_24,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_244 = torch.nn.functional.batch_norm(
            x_243,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_243 = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_242 += x_244
        out_122 = x_242
        x_242 = x_244 = None
        x_245 = torch.conv2d(
            input_24,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_246 = torch.nn.functional.batch_norm(
            x_245,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_245 = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_122 += x_246
        out_123 = out_122
        out_122 = x_246 = None
        x_247 = torch.conv2d(
            input_24,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_248 = torch.nn.functional.batch_norm(
            x_247,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_247 = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_123 += x_248
        out_124 = out_123
        out_123 = x_248 = None
        x_249 = torch.conv2d(
            input_24,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        input_24 = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_250 = torch.nn.functional.batch_norm(
            x_249,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_249 = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_124 += x_250
        out_125 = out_124
        out_124 = x_250 = None
        out_125 += x_240
        out_126 = out_125
        out_125 = x_240 = None
        input_25 = torch.nn.functional.relu(out_126, inplace=True)
        out_126 = None
        x_251 = torch.nn.functional.batch_norm(
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
        x_252 = torch.conv2d(
            input_25,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_253 = torch.nn.functional.batch_norm(
            x_252,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_252 = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_127 = 0 + x_253
        x_253 = None
        x_254 = torch.conv2d(
            input_25,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_255 = torch.nn.functional.batch_norm(
            x_254,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_254 = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_127 += x_255
        out_128 = out_127
        out_127 = x_255 = None
        x_256 = torch.conv2d(
            input_25,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_257 = torch.nn.functional.batch_norm(
            x_256,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_256 = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_128 += x_257
        out_129 = out_128
        out_128 = x_257 = None
        x_258 = torch.conv2d(
            input_25,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_25 = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_259 = torch.nn.functional.batch_norm(
            x_258,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_258 = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_129 += x_259
        out_130 = out_129
        out_129 = x_259 = None
        out_130 += x_251
        out_131 = out_130
        out_130 = x_251 = None
        input_26 = torch.nn.functional.relu(out_131, inplace=True)
        out_131 = None
        x_260 = torch.nn.functional.batch_norm(
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
        x_261 = torch.conv2d(
            input_26,
            l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_262 = torch.nn.functional.batch_norm(
            x_261,
            l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_261 = l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_263 = torch.conv2d(
            input_26,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_264 = torch.nn.functional.batch_norm(
            x_263,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_263 = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_262 += x_264
        out_132 = x_262
        x_262 = x_264 = None
        x_265 = torch.conv2d(
            input_26,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_266 = torch.nn.functional.batch_norm(
            x_265,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_265 = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_132 += x_266
        out_133 = out_132
        out_132 = x_266 = None
        x_267 = torch.conv2d(
            input_26,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_268 = torch.nn.functional.batch_norm(
            x_267,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_267 = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_133 += x_268
        out_134 = out_133
        out_133 = x_268 = None
        x_269 = torch.conv2d(
            input_26,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        input_26 = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_270 = torch.nn.functional.batch_norm(
            x_269,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_269 = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_134 += x_270
        out_135 = out_134
        out_134 = x_270 = None
        out_135 += x_260
        out_136 = out_135
        out_135 = x_260 = None
        input_27 = torch.nn.functional.relu(out_136, inplace=True)
        out_136 = None
        x_271 = torch.nn.functional.batch_norm(
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
        x_272 = torch.conv2d(
            input_27,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_273 = torch.nn.functional.batch_norm(
            x_272,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_272 = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_137 = 0 + x_273
        x_273 = None
        x_274 = torch.conv2d(
            input_27,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_275 = torch.nn.functional.batch_norm(
            x_274,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_274 = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_137 += x_275
        out_138 = out_137
        out_137 = x_275 = None
        x_276 = torch.conv2d(
            input_27,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_277 = torch.nn.functional.batch_norm(
            x_276,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_276 = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_138 += x_277
        out_139 = out_138
        out_138 = x_277 = None
        x_278 = torch.conv2d(
            input_27,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_27 = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_279 = torch.nn.functional.batch_norm(
            x_278,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_278 = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_7_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_139 += x_279
        out_140 = out_139
        out_139 = x_279 = None
        out_140 += x_271
        out_141 = out_140
        out_140 = x_271 = None
        input_28 = torch.nn.functional.relu(out_141, inplace=True)
        out_141 = None
        x_280 = torch.nn.functional.batch_norm(
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
        x_281 = torch.conv2d(
            input_28,
            l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_282 = torch.nn.functional.batch_norm(
            x_281,
            l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_281 = l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_283 = torch.conv2d(
            input_28,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_284 = torch.nn.functional.batch_norm(
            x_283,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_283 = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_282 += x_284
        out_142 = x_282
        x_282 = x_284 = None
        x_285 = torch.conv2d(
            input_28,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_286 = torch.nn.functional.batch_norm(
            x_285,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_285 = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_142 += x_286
        out_143 = out_142
        out_142 = x_286 = None
        x_287 = torch.conv2d(
            input_28,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_288 = torch.nn.functional.batch_norm(
            x_287,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_287 = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_143 += x_288
        out_144 = out_143
        out_143 = x_288 = None
        x_289 = torch.conv2d(
            input_28,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        input_28 = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_290 = torch.nn.functional.batch_norm(
            x_289,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_289 = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_8_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_144 += x_290
        out_145 = out_144
        out_144 = x_290 = None
        out_145 += x_280
        out_146 = out_145
        out_145 = x_280 = None
        input_29 = torch.nn.functional.relu(out_146, inplace=True)
        out_146 = None
        x_291 = torch.nn.functional.batch_norm(
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
        x_292 = torch.conv2d(
            input_29,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_293 = torch.nn.functional.batch_norm(
            x_292,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_292 = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_147 = 0 + x_293
        x_293 = None
        x_294 = torch.conv2d(
            input_29,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_295 = torch.nn.functional.batch_norm(
            x_294,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_294 = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_147 += x_295
        out_148 = out_147
        out_147 = x_295 = None
        x_296 = torch.conv2d(
            input_29,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_297 = torch.nn.functional.batch_norm(
            x_296,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_296 = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_148 += x_297
        out_149 = out_148
        out_148 = x_297 = None
        x_298 = torch.conv2d(
            input_29,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_29 = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_299 = torch.nn.functional.batch_norm(
            x_298,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_298 = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_9_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_149 += x_299
        out_150 = out_149
        out_149 = x_299 = None
        out_150 += x_291
        out_151 = out_150
        out_150 = x_291 = None
        input_30 = torch.nn.functional.relu(out_151, inplace=True)
        out_151 = None
        x_300 = torch.nn.functional.batch_norm(
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
        x_301 = torch.conv2d(
            input_30,
            l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_302 = torch.nn.functional.batch_norm(
            x_301,
            l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_301 = l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_303 = torch.conv2d(
            input_30,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_304 = torch.nn.functional.batch_norm(
            x_303,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_303 = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_302 += x_304
        out_152 = x_302
        x_302 = x_304 = None
        x_305 = torch.conv2d(
            input_30,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_306 = torch.nn.functional.batch_norm(
            x_305,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_305 = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_152 += x_306
        out_153 = out_152
        out_152 = x_306 = None
        x_307 = torch.conv2d(
            input_30,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_308 = torch.nn.functional.batch_norm(
            x_307,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_307 = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_153 += x_308
        out_154 = out_153
        out_153 = x_308 = None
        x_309 = torch.conv2d(
            input_30,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        input_30 = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_310 = torch.nn.functional.batch_norm(
            x_309,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_309 = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_10_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_154 += x_310
        out_155 = out_154
        out_154 = x_310 = None
        out_155 += x_300
        out_156 = out_155
        out_155 = x_300 = None
        input_31 = torch.nn.functional.relu(out_156, inplace=True)
        out_156 = None
        x_311 = torch.nn.functional.batch_norm(
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
        x_312 = torch.conv2d(
            input_31,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_313 = torch.nn.functional.batch_norm(
            x_312,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_312 = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_157 = 0 + x_313
        x_313 = None
        x_314 = torch.conv2d(
            input_31,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_315 = torch.nn.functional.batch_norm(
            x_314,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_314 = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_157 += x_315
        out_158 = out_157
        out_157 = x_315 = None
        x_316 = torch.conv2d(
            input_31,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_317 = torch.nn.functional.batch_norm(
            x_316,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_316 = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_158 += x_317
        out_159 = out_158
        out_158 = x_317 = None
        x_318 = torch.conv2d(
            input_31,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_31 = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_319 = torch.nn.functional.batch_norm(
            x_318,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_318 = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_11_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_159 += x_319
        out_160 = out_159
        out_159 = x_319 = None
        out_160 += x_311
        out_161 = out_160
        out_160 = x_311 = None
        input_32 = torch.nn.functional.relu(out_161, inplace=True)
        out_161 = None
        x_320 = torch.nn.functional.batch_norm(
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
        x_321 = torch.conv2d(
            input_32,
            l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_322 = torch.nn.functional.batch_norm(
            x_321,
            l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_321 = l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_323 = torch.conv2d(
            input_32,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_324 = torch.nn.functional.batch_norm(
            x_323,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_323 = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_322 += x_324
        out_162 = x_322
        x_322 = x_324 = None
        x_325 = torch.conv2d(
            input_32,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_326 = torch.nn.functional.batch_norm(
            x_325,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_325 = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_162 += x_326
        out_163 = out_162
        out_162 = x_326 = None
        x_327 = torch.conv2d(
            input_32,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_328 = torch.nn.functional.batch_norm(
            x_327,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_327 = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_163 += x_328
        out_164 = out_163
        out_163 = x_328 = None
        x_329 = torch.conv2d(
            input_32,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        input_32 = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_330 = torch.nn.functional.batch_norm(
            x_329,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_329 = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_12_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_164 += x_330
        out_165 = out_164
        out_164 = x_330 = None
        out_165 += x_320
        out_166 = out_165
        out_165 = x_320 = None
        input_33 = torch.nn.functional.relu(out_166, inplace=True)
        out_166 = None
        x_331 = torch.nn.functional.batch_norm(
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
        x_332 = torch.conv2d(
            input_33,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_333 = torch.nn.functional.batch_norm(
            x_332,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_332 = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_167 = 0 + x_333
        x_333 = None
        x_334 = torch.conv2d(
            input_33,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_335 = torch.nn.functional.batch_norm(
            x_334,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_334 = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_167 += x_335
        out_168 = out_167
        out_167 = x_335 = None
        x_336 = torch.conv2d(
            input_33,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_337 = torch.nn.functional.batch_norm(
            x_336,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_336 = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_168 += x_337
        out_169 = out_168
        out_168 = x_337 = None
        x_338 = torch.conv2d(
            input_33,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_33 = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_339 = torch.nn.functional.batch_norm(
            x_338,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_338 = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_13_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_169 += x_339
        out_170 = out_169
        out_169 = x_339 = None
        out_170 += x_331
        out_171 = out_170
        out_170 = x_331 = None
        input_34 = torch.nn.functional.relu(out_171, inplace=True)
        out_171 = None
        x_340 = torch.nn.functional.batch_norm(
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
        x_341 = torch.conv2d(
            input_34,
            l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_342 = torch.nn.functional.batch_norm(
            x_341,
            l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_341 = l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_343 = torch.conv2d(
            input_34,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_344 = torch.nn.functional.batch_norm(
            x_343,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_343 = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_342 += x_344
        out_172 = x_342
        x_342 = x_344 = None
        x_345 = torch.conv2d(
            input_34,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_346 = torch.nn.functional.batch_norm(
            x_345,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_345 = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_172 += x_346
        out_173 = out_172
        out_172 = x_346 = None
        x_347 = torch.conv2d(
            input_34,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_348 = torch.nn.functional.batch_norm(
            x_347,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_347 = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_173 += x_348
        out_174 = out_173
        out_173 = x_348 = None
        x_349 = torch.conv2d(
            input_34,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        input_34 = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_350 = torch.nn.functional.batch_norm(
            x_349,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_349 = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_14_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_174 += x_350
        out_175 = out_174
        out_174 = x_350 = None
        out_175 += x_340
        out_176 = out_175
        out_175 = x_340 = None
        input_35 = torch.nn.functional.relu(out_176, inplace=True)
        out_176 = None
        x_351 = torch.nn.functional.batch_norm(
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
        x_352 = torch.conv2d(
            input_35,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_353 = torch.nn.functional.batch_norm(
            x_352,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_352 = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_177 = 0 + x_353
        x_353 = None
        x_354 = torch.conv2d(
            input_35,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_355 = torch.nn.functional.batch_norm(
            x_354,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_354 = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_177 += x_355
        out_178 = out_177
        out_177 = x_355 = None
        x_356 = torch.conv2d(
            input_35,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_357 = torch.nn.functional.batch_norm(
            x_356,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_356 = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_178 += x_357
        out_179 = out_178
        out_178 = x_357 = None
        x_358 = torch.conv2d(
            input_35,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_35 = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_359 = torch.nn.functional.batch_norm(
            x_358,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_358 = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_15_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_179 += x_359
        out_180 = out_179
        out_179 = x_359 = None
        out_180 += x_351
        out_181 = out_180
        out_180 = x_351 = None
        input_36 = torch.nn.functional.relu(out_181, inplace=True)
        out_181 = None
        x_360 = torch.nn.functional.batch_norm(
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
        x_361 = torch.conv2d(
            input_36,
            l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_362 = torch.nn.functional.batch_norm(
            x_361,
            l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_361 = l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_363 = torch.conv2d(
            input_36,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_364 = torch.nn.functional.batch_norm(
            x_363,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_363 = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_362 += x_364
        out_182 = x_362
        x_362 = x_364 = None
        x_365 = torch.conv2d(
            input_36,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_366 = torch.nn.functional.batch_norm(
            x_365,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_365 = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_182 += x_366
        out_183 = out_182
        out_182 = x_366 = None
        x_367 = torch.conv2d(
            input_36,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_368 = torch.nn.functional.batch_norm(
            x_367,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_367 = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_183 += x_368
        out_184 = out_183
        out_183 = x_368 = None
        x_369 = torch.conv2d(
            input_36,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        input_36 = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_370 = torch.nn.functional.batch_norm(
            x_369,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_369 = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_16_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_184 += x_370
        out_185 = out_184
        out_184 = x_370 = None
        out_185 += x_360
        out_186 = out_185
        out_185 = x_360 = None
        input_37 = torch.nn.functional.relu(out_186, inplace=True)
        out_186 = None
        x_371 = torch.nn.functional.batch_norm(
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
        x_372 = torch.conv2d(
            input_37,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_373 = torch.nn.functional.batch_norm(
            x_372,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_372 = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_187 = 0 + x_373
        x_373 = None
        x_374 = torch.conv2d(
            input_37,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_375 = torch.nn.functional.batch_norm(
            x_374,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_374 = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_187 += x_375
        out_188 = out_187
        out_187 = x_375 = None
        x_376 = torch.conv2d(
            input_37,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_377 = torch.nn.functional.batch_norm(
            x_376,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_376 = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_188 += x_377
        out_189 = out_188
        out_188 = x_377 = None
        x_378 = torch.conv2d(
            input_37,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_37 = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_379 = torch.nn.functional.batch_norm(
            x_378,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_378 = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_17_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_189 += x_379
        out_190 = out_189
        out_189 = x_379 = None
        out_190 += x_371
        out_191 = out_190
        out_190 = x_371 = None
        input_38 = torch.nn.functional.relu(out_191, inplace=True)
        out_191 = None
        x_380 = torch.nn.functional.batch_norm(
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
        x_381 = torch.conv2d(
            input_38,
            l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_382 = torch.nn.functional.batch_norm(
            x_381,
            l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_381 = l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_383 = torch.conv2d(
            input_38,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_384 = torch.nn.functional.batch_norm(
            x_383,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_383 = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_382 += x_384
        out_192 = x_382
        x_382 = x_384 = None
        x_385 = torch.conv2d(
            input_38,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_386 = torch.nn.functional.batch_norm(
            x_385,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_385 = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_192 += x_386
        out_193 = out_192
        out_192 = x_386 = None
        x_387 = torch.conv2d(
            input_38,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_388 = torch.nn.functional.batch_norm(
            x_387,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_387 = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_193 += x_388
        out_194 = out_193
        out_193 = x_388 = None
        x_389 = torch.conv2d(
            input_38,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        input_38 = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_390 = torch.nn.functional.batch_norm(
            x_389,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_389 = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_18_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_194 += x_390
        out_195 = out_194
        out_194 = x_390 = None
        out_195 += x_380
        out_196 = out_195
        out_195 = x_380 = None
        input_39 = torch.nn.functional.relu(out_196, inplace=True)
        out_196 = None
        x_391 = torch.nn.functional.batch_norm(
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
        x_392 = torch.conv2d(
            input_39,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_393 = torch.nn.functional.batch_norm(
            x_392,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_392 = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_197 = 0 + x_393
        x_393 = None
        x_394 = torch.conv2d(
            input_39,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_395 = torch.nn.functional.batch_norm(
            x_394,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_394 = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_197 += x_395
        out_198 = out_197
        out_197 = x_395 = None
        x_396 = torch.conv2d(
            input_39,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_397 = torch.nn.functional.batch_norm(
            x_396,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_396 = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_198 += x_397
        out_199 = out_198
        out_198 = x_397 = None
        x_398 = torch.conv2d(
            input_39,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_39 = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_399 = torch.nn.functional.batch_norm(
            x_398,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_398 = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_19_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_199 += x_399
        out_200 = out_199
        out_199 = x_399 = None
        out_200 += x_391
        out_201 = out_200
        out_200 = x_391 = None
        input_40 = torch.nn.functional.relu(out_201, inplace=True)
        out_201 = None
        x_400 = torch.conv2d(
            input_40,
            l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_conv_parameters_weight_ = (
            None
        )
        x_401 = torch.nn.functional.batch_norm(
            x_400,
            l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_400 = l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv_scale_modules_bn_parameters_bias_ = (None)
        x_402 = torch.conv2d(
            input_40,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_403 = torch.nn.functional.batch_norm(
            x_402,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_402 = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        x_401 += x_403
        out_202 = x_401
        x_401 = x_403 = None
        x_404 = torch.conv2d(
            input_40,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_405 = torch.nn.functional.batch_norm(
            x_404,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_404 = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_202 += x_405
        out_203 = out_202
        out_202 = x_405 = None
        x_406 = torch.conv2d(
            input_40,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_407 = torch.nn.functional.batch_norm(
            x_406,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_406 = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_203 += x_407
        out_204 = out_203
        out_203 = x_407 = None
        x_408 = torch.conv2d(
            input_40,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            256,
        )
        input_40 = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_409 = torch.nn.functional.batch_norm(
            x_408,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_408 = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_0_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_204 += x_409
        out_205 = out_204
        out_204 = x_409 = None
        out_205 += 0
        out_206 = out_205
        out_205 = None
        input_41 = torch.nn.functional.relu(out_206, inplace=True)
        out_206 = None
        x_410 = torch.conv2d(
            input_41,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_411 = torch.nn.functional.batch_norm(
            x_410,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_410 = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_0_modules_bn_parameters_bias_ = (None)
        out_207 = 0 + x_411
        x_411 = None
        x_412 = torch.conv2d(
            input_41,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_conv_parameters_weight_ = (
            None
        )
        x_413 = torch.nn.functional.batch_norm(
            x_412,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_412 = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_1_modules_bn_parameters_bias_ = (None)
        out_207 += x_413
        out_208 = out_207
        out_207 = x_413 = None
        x_414 = torch.conv2d(
            input_41,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_conv_parameters_weight_ = (
            None
        )
        x_415 = torch.nn.functional.batch_norm(
            x_414,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_414 = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_2_modules_bn_parameters_bias_ = (None)
        out_208 += x_415
        out_209 = out_208
        out_208 = x_415 = None
        x_416 = torch.conv2d(
            input_41,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_41 = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_conv_parameters_weight_ = (None)
        x_417 = torch.nn.functional.batch_norm(
            x_416,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_416 = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_conv_kxk_modules_3_modules_bn_parameters_bias_ = (None)
        out_209 += x_417
        out_210 = out_209
        out_209 = x_417 = None
        out_210 += 0
        out_211 = out_210
        out_210 = None
        input_42 = torch.nn.functional.relu(out_211, inplace=True)
        out_211 = None
        x_418 = torch.nn.functional.adaptive_avg_pool2d(input_42, 1)
        input_42 = None
        x_419 = x_418.flatten(1, -1)
        x_418 = None
        x_420 = torch.nn.functional.dropout(x_419, 0.0, False, False)
        x_419 = None
        x_421 = torch._C._nn.linear(
            x_420,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_420 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_421,)
