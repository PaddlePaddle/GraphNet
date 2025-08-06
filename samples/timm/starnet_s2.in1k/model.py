import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_f1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_f1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_f2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_f2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_g_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_g_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_dwconv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_1_modules_dwconv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_f1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_f1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_f2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_f2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_g_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_g_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_dwconv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_1_modules_dwconv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_f1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_f1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_f2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_f2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_g_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_g_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_dwconv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_2_modules_dwconv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_f1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_f1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_f2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_f2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_g_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_g_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_dwconv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_1_modules_dwconv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_f1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_f1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_f2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_f2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_g_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_g_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_dwconv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_2_modules_dwconv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_f1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_f1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_f2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_f2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_g_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_g_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_dwconv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_3_modules_dwconv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_f1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_f1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_f2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_f2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_g_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_g_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_dwconv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_4_modules_dwconv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_f1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_f1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_f2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_f2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_g_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_g_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_dwconv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_5_modules_dwconv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_f1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_f1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_f2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_f2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_g_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_g_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_dwconv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_6_modules_dwconv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_f1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_f1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_f2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_f2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_g_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_g_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_dwconv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_1_modules_dwconv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_f1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_f1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_f2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_f2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_g_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_g_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_dwconv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_2_modules_dwconv2_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_0_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_0_modules_conv_parameters_bias_ = (
            L_self_modules_stem_modules_0_modules_conv_parameters_bias_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_0_modules_conv_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_0_modules_conv_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_0_modules_conv_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_stages_modules_0_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_stages_modules_0_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_stages_modules_0_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_stages_modules_0_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_stages_modules_0_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_stages_modules_0_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_conv_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_conv_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_f1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_f1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_f1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_f1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_f2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_f2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_f2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_f2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_g_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_g_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_g_modules_conv_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_g_modules_conv_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_1_modules_dwconv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_1_modules_dwconv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_1_modules_dwconv2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_0_modules_1_modules_dwconv2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_0_modules_conv_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_0_modules_conv_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_0_modules_conv_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_stages_modules_1_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_stages_modules_1_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_stages_modules_1_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_stages_modules_1_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_stages_modules_1_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_stages_modules_1_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_f1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_f1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_f1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_f1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_f2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_f2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_f2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_f2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_g_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_g_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_g_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_g_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_1_modules_dwconv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_1_modules_dwconv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_1_modules_dwconv2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_1_modules_dwconv2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_f1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_f1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_f1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_f1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_f2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_f2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_f2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_f2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_g_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_g_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_g_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_g_modules_conv_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_2_modules_dwconv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_2_modules_dwconv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_2_modules_dwconv2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_1_modules_2_modules_dwconv2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_0_modules_conv_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_0_modules_conv_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_0_modules_conv_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_stages_modules_2_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_stages_modules_2_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_stages_modules_2_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_stages_modules_2_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_stages_modules_2_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_stages_modules_2_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_f1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_f1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_f1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_f1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_f2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_f2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_f2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_f2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_g_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_g_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_g_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_g_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_1_modules_dwconv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_1_modules_dwconv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_1_modules_dwconv2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_1_modules_dwconv2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_f1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_f1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_f1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_f1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_f2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_f2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_f2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_f2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_g_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_g_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_g_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_g_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_2_modules_dwconv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_2_modules_dwconv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_2_modules_dwconv2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_2_modules_dwconv2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_f1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_f1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_f1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_f1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_f2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_f2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_f2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_f2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_g_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_g_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_g_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_g_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_3_modules_dwconv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_3_modules_dwconv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_3_modules_dwconv2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_3_modules_dwconv2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_f1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_f1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_f1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_f1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_f2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_f2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_f2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_f2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_g_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_g_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_g_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_g_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_4_modules_dwconv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_4_modules_dwconv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_4_modules_dwconv2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_4_modules_dwconv2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_f1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_f1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_f1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_f1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_f2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_f2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_f2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_f2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_g_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_g_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_g_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_g_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_5_modules_dwconv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_5_modules_dwconv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_5_modules_dwconv2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_5_modules_dwconv2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_f1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_f1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_f1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_f1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_f2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_f2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_f2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_f2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_g_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_g_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_g_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_g_modules_conv_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_6_modules_dwconv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_6_modules_dwconv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_6_modules_dwconv2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_2_modules_6_modules_dwconv2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_0_modules_conv_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_0_modules_conv_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_0_modules_conv_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_0_modules_bn_buffers_running_mean_ = (
            L_self_modules_stages_modules_3_modules_0_modules_bn_buffers_running_mean_
        )
        l_self_modules_stages_modules_3_modules_0_modules_bn_buffers_running_var_ = (
            L_self_modules_stages_modules_3_modules_0_modules_bn_buffers_running_var_
        )
        l_self_modules_stages_modules_3_modules_0_modules_bn_parameters_weight_ = (
            L_self_modules_stages_modules_3_modules_0_modules_bn_parameters_weight_
        )
        l_self_modules_stages_modules_3_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_f1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_f1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_f1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_f1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_f2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_f2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_f2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_f2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_g_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_g_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_g_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_g_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_1_modules_dwconv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_1_modules_dwconv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_1_modules_dwconv2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_1_modules_dwconv2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_2_modules_f1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_2_modules_f1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_2_modules_f1_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_2_modules_f1_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_2_modules_f2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_2_modules_f2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_2_modules_f2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_2_modules_f2_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_2_modules_g_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_2_modules_g_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_2_modules_g_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_2_modules_g_modules_conv_parameters_bias_
        l_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_2_modules_dwconv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_2_modules_dwconv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_2_modules_dwconv2_modules_conv_parameters_bias_ = L_self_modules_stages_modules_3_modules_2_modules_dwconv2_modules_conv_parameters_bias_
        l_self_modules_norm_buffers_running_mean_ = (
            L_self_modules_norm_buffers_running_mean_
        )
        l_self_modules_norm_buffers_running_var_ = (
            L_self_modules_norm_buffers_running_var_
        )
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
        l_self_modules_norm_parameters_bias_ = L_self_modules_norm_parameters_bias_
        l_self_modules_head_parameters_weight_ = L_self_modules_head_parameters_weight_
        l_self_modules_head_parameters_bias_ = L_self_modules_head_parameters_bias_
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_0_modules_conv_parameters_weight_,
            l_self_modules_stem_modules_0_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_stem_modules_0_modules_conv_parameters_weight_
        ) = l_self_modules_stem_modules_0_modules_conv_parameters_bias_ = None
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_stem_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = (
            l_self_modules_stem_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_0_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_0_modules_bn_parameters_bias_ = None
        input_3 = torch.nn.functional.hardtanh(input_2, 0.0, 6.0, False)
        input_2 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_0_modules_0_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_3 = (
            l_self_modules_stages_modules_0_modules_0_modules_conv_parameters_weight_
        ) = (
            l_self_modules_stages_modules_0_modules_0_modules_conv_parameters_bias_
        ) = None
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_stages_modules_0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_4 = (
            l_self_modules_stages_modules_0_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stages_modules_0_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stages_modules_0_modules_0_modules_bn_parameters_weight_
        ) = l_self_modules_stages_modules_0_modules_0_modules_bn_parameters_bias_ = None
        input_6 = torch.conv2d(
            input_5,
            l_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            32,
        )
        l_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_conv_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_conv_parameters_bias_ = (None)
        input_7 = torch.nn.functional.batch_norm(
            input_6,
            l_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_6 = l_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_dwconv_modules_bn_parameters_bias_ = (None)
        input_8 = torch.conv2d(
            input_7,
            l_self_modules_stages_modules_0_modules_1_modules_f1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_f1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_1_modules_f1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_f1_modules_conv_parameters_bias_ = (None)
        input_9 = torch.conv2d(
            input_7,
            l_self_modules_stages_modules_0_modules_1_modules_f2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_f2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_7 = l_self_modules_stages_modules_0_modules_1_modules_f2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_f2_modules_conv_parameters_bias_ = (None)
        hardtanh_1 = torch.nn.functional.hardtanh(input_8, 0.0, 6.0, False)
        input_8 = None
        x = hardtanh_1 * input_9
        hardtanh_1 = input_9 = None
        input_10 = torch.conv2d(
            x,
            l_self_modules_stages_modules_0_modules_1_modules_g_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_g_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x = l_self_modules_stages_modules_0_modules_1_modules_g_modules_conv_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_g_modules_conv_parameters_bias_ = (None)
        input_11 = torch.nn.functional.batch_norm(
            input_10,
            l_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_10 = l_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_g_modules_bn_parameters_bias_ = (None)
        input_12 = torch.conv2d(
            input_11,
            l_self_modules_stages_modules_0_modules_1_modules_dwconv2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_0_modules_1_modules_dwconv2_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            32,
        )
        input_11 = l_self_modules_stages_modules_0_modules_1_modules_dwconv2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_0_modules_1_modules_dwconv2_modules_conv_parameters_bias_ = (None)
        x_1 = input_5 + input_12
        input_5 = input_12 = None
        input_13 = torch.conv2d(
            x_1,
            l_self_modules_stages_modules_1_modules_0_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_1 = (
            l_self_modules_stages_modules_1_modules_0_modules_conv_parameters_weight_
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_conv_parameters_bias_
        ) = None
        input_14 = torch.nn.functional.batch_norm(
            input_13,
            l_self_modules_stages_modules_1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_13 = (
            l_self_modules_stages_modules_1_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stages_modules_1_modules_0_modules_bn_parameters_weight_
        ) = l_self_modules_stages_modules_1_modules_0_modules_bn_parameters_bias_ = None
        input_15 = torch.conv2d(
            input_14,
            l_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            64,
        )
        l_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_conv_parameters_bias_ = (None)
        input_16 = torch.nn.functional.batch_norm(
            input_15,
            l_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_15 = l_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_dwconv_modules_bn_parameters_bias_ = (None)
        input_17 = torch.conv2d(
            input_16,
            l_self_modules_stages_modules_1_modules_1_modules_f1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_f1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_1_modules_f1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_f1_modules_conv_parameters_bias_ = (None)
        input_18 = torch.conv2d(
            input_16,
            l_self_modules_stages_modules_1_modules_1_modules_f2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_f2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_16 = l_self_modules_stages_modules_1_modules_1_modules_f2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_f2_modules_conv_parameters_bias_ = (None)
        hardtanh_2 = torch.nn.functional.hardtanh(input_17, 0.0, 6.0, False)
        input_17 = None
        x_2 = hardtanh_2 * input_18
        hardtanh_2 = input_18 = None
        input_19 = torch.conv2d(
            x_2,
            l_self_modules_stages_modules_1_modules_1_modules_g_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_g_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_stages_modules_1_modules_1_modules_g_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_g_modules_conv_parameters_bias_ = (None)
        input_20 = torch.nn.functional.batch_norm(
            input_19,
            l_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_19 = l_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_g_modules_bn_parameters_bias_ = (None)
        input_21 = torch.conv2d(
            input_20,
            l_self_modules_stages_modules_1_modules_1_modules_dwconv2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_1_modules_dwconv2_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            64,
        )
        input_20 = l_self_modules_stages_modules_1_modules_1_modules_dwconv2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_1_modules_dwconv2_modules_conv_parameters_bias_ = (None)
        x_3 = input_14 + input_21
        input_14 = input_21 = None
        input_22 = torch.conv2d(
            x_3,
            l_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            64,
        )
        l_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_conv_parameters_bias_ = (None)
        input_23 = torch.nn.functional.batch_norm(
            input_22,
            l_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_22 = l_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_dwconv_modules_bn_parameters_bias_ = (None)
        input_24 = torch.conv2d(
            input_23,
            l_self_modules_stages_modules_1_modules_2_modules_f1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_f1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_2_modules_f1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_f1_modules_conv_parameters_bias_ = (None)
        input_25 = torch.conv2d(
            input_23,
            l_self_modules_stages_modules_1_modules_2_modules_f2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_f2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_23 = l_self_modules_stages_modules_1_modules_2_modules_f2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_f2_modules_conv_parameters_bias_ = (None)
        hardtanh_3 = torch.nn.functional.hardtanh(input_24, 0.0, 6.0, False)
        input_24 = None
        x_4 = hardtanh_3 * input_25
        hardtanh_3 = input_25 = None
        input_26 = torch.conv2d(
            x_4,
            l_self_modules_stages_modules_1_modules_2_modules_g_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_g_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_4 = l_self_modules_stages_modules_1_modules_2_modules_g_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_g_modules_conv_parameters_bias_ = (None)
        input_27 = torch.nn.functional.batch_norm(
            input_26,
            l_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_26 = l_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_g_modules_bn_parameters_bias_ = (None)
        input_28 = torch.conv2d(
            input_27,
            l_self_modules_stages_modules_1_modules_2_modules_dwconv2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_1_modules_2_modules_dwconv2_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            64,
        )
        input_27 = l_self_modules_stages_modules_1_modules_2_modules_dwconv2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_1_modules_2_modules_dwconv2_modules_conv_parameters_bias_ = (None)
        x_5 = x_3 + input_28
        x_3 = input_28 = None
        input_29 = torch.conv2d(
            x_5,
            l_self_modules_stages_modules_2_modules_0_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_5 = (
            l_self_modules_stages_modules_2_modules_0_modules_conv_parameters_weight_
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_conv_parameters_bias_
        ) = None
        input_30 = torch.nn.functional.batch_norm(
            input_29,
            l_self_modules_stages_modules_2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_29 = (
            l_self_modules_stages_modules_2_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stages_modules_2_modules_0_modules_bn_parameters_weight_
        ) = l_self_modules_stages_modules_2_modules_0_modules_bn_parameters_bias_ = None
        input_31 = torch.conv2d(
            input_30,
            l_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_conv_parameters_bias_ = (None)
        input_32 = torch.nn.functional.batch_norm(
            input_31,
            l_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_31 = l_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_dwconv_modules_bn_parameters_bias_ = (None)
        input_33 = torch.conv2d(
            input_32,
            l_self_modules_stages_modules_2_modules_1_modules_f1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_f1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_1_modules_f1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_f1_modules_conv_parameters_bias_ = (None)
        input_34 = torch.conv2d(
            input_32,
            l_self_modules_stages_modules_2_modules_1_modules_f2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_f2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_32 = l_self_modules_stages_modules_2_modules_1_modules_f2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_f2_modules_conv_parameters_bias_ = (None)
        hardtanh_4 = torch.nn.functional.hardtanh(input_33, 0.0, 6.0, False)
        input_33 = None
        x_6 = hardtanh_4 * input_34
        hardtanh_4 = input_34 = None
        input_35 = torch.conv2d(
            x_6,
            l_self_modules_stages_modules_2_modules_1_modules_g_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_g_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_6 = l_self_modules_stages_modules_2_modules_1_modules_g_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_g_modules_conv_parameters_bias_ = (None)
        input_36 = torch.nn.functional.batch_norm(
            input_35,
            l_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_35 = l_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_g_modules_bn_parameters_bias_ = (None)
        input_37 = torch.conv2d(
            input_36,
            l_self_modules_stages_modules_2_modules_1_modules_dwconv2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_1_modules_dwconv2_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        input_36 = l_self_modules_stages_modules_2_modules_1_modules_dwconv2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_1_modules_dwconv2_modules_conv_parameters_bias_ = (None)
        x_7 = input_30 + input_37
        input_30 = input_37 = None
        input_38 = torch.conv2d(
            x_7,
            l_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_conv_parameters_bias_ = (None)
        input_39 = torch.nn.functional.batch_norm(
            input_38,
            l_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_38 = l_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_dwconv_modules_bn_parameters_bias_ = (None)
        input_40 = torch.conv2d(
            input_39,
            l_self_modules_stages_modules_2_modules_2_modules_f1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_f1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_2_modules_f1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_f1_modules_conv_parameters_bias_ = (None)
        input_41 = torch.conv2d(
            input_39,
            l_self_modules_stages_modules_2_modules_2_modules_f2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_f2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_39 = l_self_modules_stages_modules_2_modules_2_modules_f2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_f2_modules_conv_parameters_bias_ = (None)
        hardtanh_5 = torch.nn.functional.hardtanh(input_40, 0.0, 6.0, False)
        input_40 = None
        x_8 = hardtanh_5 * input_41
        hardtanh_5 = input_41 = None
        input_42 = torch.conv2d(
            x_8,
            l_self_modules_stages_modules_2_modules_2_modules_g_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_g_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_8 = l_self_modules_stages_modules_2_modules_2_modules_g_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_g_modules_conv_parameters_bias_ = (None)
        input_43 = torch.nn.functional.batch_norm(
            input_42,
            l_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_42 = l_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_g_modules_bn_parameters_bias_ = (None)
        input_44 = torch.conv2d(
            input_43,
            l_self_modules_stages_modules_2_modules_2_modules_dwconv2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_2_modules_dwconv2_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        input_43 = l_self_modules_stages_modules_2_modules_2_modules_dwconv2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_2_modules_dwconv2_modules_conv_parameters_bias_ = (None)
        x_9 = x_7 + input_44
        x_7 = input_44 = None
        input_45 = torch.conv2d(
            x_9,
            l_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_conv_parameters_bias_ = (None)
        input_46 = torch.nn.functional.batch_norm(
            input_45,
            l_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_45 = l_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_dwconv_modules_bn_parameters_bias_ = (None)
        input_47 = torch.conv2d(
            input_46,
            l_self_modules_stages_modules_2_modules_3_modules_f1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_f1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_3_modules_f1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_f1_modules_conv_parameters_bias_ = (None)
        input_48 = torch.conv2d(
            input_46,
            l_self_modules_stages_modules_2_modules_3_modules_f2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_f2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_46 = l_self_modules_stages_modules_2_modules_3_modules_f2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_f2_modules_conv_parameters_bias_ = (None)
        hardtanh_6 = torch.nn.functional.hardtanh(input_47, 0.0, 6.0, False)
        input_47 = None
        x_10 = hardtanh_6 * input_48
        hardtanh_6 = input_48 = None
        input_49 = torch.conv2d(
            x_10,
            l_self_modules_stages_modules_2_modules_3_modules_g_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_g_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_10 = l_self_modules_stages_modules_2_modules_3_modules_g_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_g_modules_conv_parameters_bias_ = (None)
        input_50 = torch.nn.functional.batch_norm(
            input_49,
            l_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_49 = l_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_g_modules_bn_parameters_bias_ = (None)
        input_51 = torch.conv2d(
            input_50,
            l_self_modules_stages_modules_2_modules_3_modules_dwconv2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_3_modules_dwconv2_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        input_50 = l_self_modules_stages_modules_2_modules_3_modules_dwconv2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_3_modules_dwconv2_modules_conv_parameters_bias_ = (None)
        x_11 = x_9 + input_51
        x_9 = input_51 = None
        input_52 = torch.conv2d(
            x_11,
            l_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_conv_parameters_bias_ = (None)
        input_53 = torch.nn.functional.batch_norm(
            input_52,
            l_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_52 = l_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_dwconv_modules_bn_parameters_bias_ = (None)
        input_54 = torch.conv2d(
            input_53,
            l_self_modules_stages_modules_2_modules_4_modules_f1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_f1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_4_modules_f1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_f1_modules_conv_parameters_bias_ = (None)
        input_55 = torch.conv2d(
            input_53,
            l_self_modules_stages_modules_2_modules_4_modules_f2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_f2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_53 = l_self_modules_stages_modules_2_modules_4_modules_f2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_f2_modules_conv_parameters_bias_ = (None)
        hardtanh_7 = torch.nn.functional.hardtanh(input_54, 0.0, 6.0, False)
        input_54 = None
        x_12 = hardtanh_7 * input_55
        hardtanh_7 = input_55 = None
        input_56 = torch.conv2d(
            x_12,
            l_self_modules_stages_modules_2_modules_4_modules_g_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_g_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_12 = l_self_modules_stages_modules_2_modules_4_modules_g_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_g_modules_conv_parameters_bias_ = (None)
        input_57 = torch.nn.functional.batch_norm(
            input_56,
            l_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_56 = l_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_g_modules_bn_parameters_bias_ = (None)
        input_58 = torch.conv2d(
            input_57,
            l_self_modules_stages_modules_2_modules_4_modules_dwconv2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_4_modules_dwconv2_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        input_57 = l_self_modules_stages_modules_2_modules_4_modules_dwconv2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_4_modules_dwconv2_modules_conv_parameters_bias_ = (None)
        x_13 = x_11 + input_58
        x_11 = input_58 = None
        input_59 = torch.conv2d(
            x_13,
            l_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_conv_parameters_bias_ = (None)
        input_60 = torch.nn.functional.batch_norm(
            input_59,
            l_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_59 = l_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_dwconv_modules_bn_parameters_bias_ = (None)
        input_61 = torch.conv2d(
            input_60,
            l_self_modules_stages_modules_2_modules_5_modules_f1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_f1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_5_modules_f1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_f1_modules_conv_parameters_bias_ = (None)
        input_62 = torch.conv2d(
            input_60,
            l_self_modules_stages_modules_2_modules_5_modules_f2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_f2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_60 = l_self_modules_stages_modules_2_modules_5_modules_f2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_f2_modules_conv_parameters_bias_ = (None)
        hardtanh_8 = torch.nn.functional.hardtanh(input_61, 0.0, 6.0, False)
        input_61 = None
        x_14 = hardtanh_8 * input_62
        hardtanh_8 = input_62 = None
        input_63 = torch.conv2d(
            x_14,
            l_self_modules_stages_modules_2_modules_5_modules_g_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_g_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_14 = l_self_modules_stages_modules_2_modules_5_modules_g_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_g_modules_conv_parameters_bias_ = (None)
        input_64 = torch.nn.functional.batch_norm(
            input_63,
            l_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_63 = l_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_g_modules_bn_parameters_bias_ = (None)
        input_65 = torch.conv2d(
            input_64,
            l_self_modules_stages_modules_2_modules_5_modules_dwconv2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_5_modules_dwconv2_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        input_64 = l_self_modules_stages_modules_2_modules_5_modules_dwconv2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_5_modules_dwconv2_modules_conv_parameters_bias_ = (None)
        x_15 = x_13 + input_65
        x_13 = input_65 = None
        input_66 = torch.conv2d(
            x_15,
            l_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        l_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_conv_parameters_bias_ = (None)
        input_67 = torch.nn.functional.batch_norm(
            input_66,
            l_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_66 = l_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_dwconv_modules_bn_parameters_bias_ = (None)
        input_68 = torch.conv2d(
            input_67,
            l_self_modules_stages_modules_2_modules_6_modules_f1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_f1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_6_modules_f1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_f1_modules_conv_parameters_bias_ = (None)
        input_69 = torch.conv2d(
            input_67,
            l_self_modules_stages_modules_2_modules_6_modules_f2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_f2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_67 = l_self_modules_stages_modules_2_modules_6_modules_f2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_f2_modules_conv_parameters_bias_ = (None)
        hardtanh_9 = torch.nn.functional.hardtanh(input_68, 0.0, 6.0, False)
        input_68 = None
        x_16 = hardtanh_9 * input_69
        hardtanh_9 = input_69 = None
        input_70 = torch.conv2d(
            x_16,
            l_self_modules_stages_modules_2_modules_6_modules_g_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_g_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_16 = l_self_modules_stages_modules_2_modules_6_modules_g_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_g_modules_conv_parameters_bias_ = (None)
        input_71 = torch.nn.functional.batch_norm(
            input_70,
            l_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_70 = l_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_g_modules_bn_parameters_bias_ = (None)
        input_72 = torch.conv2d(
            input_71,
            l_self_modules_stages_modules_2_modules_6_modules_dwconv2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_2_modules_6_modules_dwconv2_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            128,
        )
        input_71 = l_self_modules_stages_modules_2_modules_6_modules_dwconv2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_2_modules_6_modules_dwconv2_modules_conv_parameters_bias_ = (None)
        x_17 = x_15 + input_72
        x_15 = input_72 = None
        input_73 = torch.conv2d(
            x_17,
            l_self_modules_stages_modules_3_modules_0_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_17 = (
            l_self_modules_stages_modules_3_modules_0_modules_conv_parameters_weight_
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_conv_parameters_bias_
        ) = None
        input_74 = torch.nn.functional.batch_norm(
            input_73,
            l_self_modules_stages_modules_3_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_0_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_73 = (
            l_self_modules_stages_modules_3_modules_0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stages_modules_3_modules_0_modules_bn_parameters_weight_
        ) = l_self_modules_stages_modules_3_modules_0_modules_bn_parameters_bias_ = None
        input_75 = torch.conv2d(
            input_74,
            l_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_conv_parameters_bias_ = (None)
        input_76 = torch.nn.functional.batch_norm(
            input_75,
            l_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_75 = l_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_dwconv_modules_bn_parameters_bias_ = (None)
        input_77 = torch.conv2d(
            input_76,
            l_self_modules_stages_modules_3_modules_1_modules_f1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_f1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_1_modules_f1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_f1_modules_conv_parameters_bias_ = (None)
        input_78 = torch.conv2d(
            input_76,
            l_self_modules_stages_modules_3_modules_1_modules_f2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_f2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_76 = l_self_modules_stages_modules_3_modules_1_modules_f2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_f2_modules_conv_parameters_bias_ = (None)
        hardtanh_10 = torch.nn.functional.hardtanh(input_77, 0.0, 6.0, False)
        input_77 = None
        x_18 = hardtanh_10 * input_78
        hardtanh_10 = input_78 = None
        input_79 = torch.conv2d(
            x_18,
            l_self_modules_stages_modules_3_modules_1_modules_g_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_g_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_stages_modules_3_modules_1_modules_g_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_g_modules_conv_parameters_bias_ = (None)
        input_80 = torch.nn.functional.batch_norm(
            input_79,
            l_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_79 = l_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_g_modules_bn_parameters_bias_ = (None)
        input_81 = torch.conv2d(
            input_80,
            l_self_modules_stages_modules_3_modules_1_modules_dwconv2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_1_modules_dwconv2_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        input_80 = l_self_modules_stages_modules_3_modules_1_modules_dwconv2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_1_modules_dwconv2_modules_conv_parameters_bias_ = (None)
        x_19 = input_74 + input_81
        input_74 = input_81 = None
        input_82 = torch.conv2d(
            x_19,
            l_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        l_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_conv_parameters_bias_ = (None)
        input_83 = torch.nn.functional.batch_norm(
            input_82,
            l_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_82 = l_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_2_modules_dwconv_modules_bn_parameters_bias_ = (None)
        input_84 = torch.conv2d(
            input_83,
            l_self_modules_stages_modules_3_modules_2_modules_f1_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_2_modules_f1_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_2_modules_f1_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_2_modules_f1_modules_conv_parameters_bias_ = (None)
        input_85 = torch.conv2d(
            input_83,
            l_self_modules_stages_modules_3_modules_2_modules_f2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_2_modules_f2_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_83 = l_self_modules_stages_modules_3_modules_2_modules_f2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_2_modules_f2_modules_conv_parameters_bias_ = (None)
        hardtanh_11 = torch.nn.functional.hardtanh(input_84, 0.0, 6.0, False)
        input_84 = None
        x_20 = hardtanh_11 * input_85
        hardtanh_11 = input_85 = None
        input_86 = torch.conv2d(
            x_20,
            l_self_modules_stages_modules_3_modules_2_modules_g_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_2_modules_g_modules_conv_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_20 = l_self_modules_stages_modules_3_modules_2_modules_g_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_2_modules_g_modules_conv_parameters_bias_ = (None)
        input_87 = torch.nn.functional.batch_norm(
            input_86,
            l_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_86 = l_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_2_modules_g_modules_bn_parameters_bias_ = (None)
        input_88 = torch.conv2d(
            input_87,
            l_self_modules_stages_modules_3_modules_2_modules_dwconv2_modules_conv_parameters_weight_,
            l_self_modules_stages_modules_3_modules_2_modules_dwconv2_modules_conv_parameters_bias_,
            (1, 1),
            (3, 3),
            (1, 1),
            256,
        )
        input_87 = l_self_modules_stages_modules_3_modules_2_modules_dwconv2_modules_conv_parameters_weight_ = l_self_modules_stages_modules_3_modules_2_modules_dwconv2_modules_conv_parameters_bias_ = (None)
        x_21 = x_19 + input_88
        x_19 = input_88 = None
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_norm_buffers_running_mean_,
            l_self_modules_norm_buffers_running_var_,
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_21 = (
            l_self_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_23 = torch.nn.functional.adaptive_avg_pool2d(x_22, 1)
        x_22 = None
        x_24 = x_23.flatten(1, -1)
        x_23 = None
        x_25 = torch._C._nn.linear(
            x_24,
            l_self_modules_head_parameters_weight_,
            l_self_modules_head_parameters_bias_,
        )
        x_24 = (
            l_self_modules_head_parameters_weight_
        ) = l_self_modules_head_parameters_bias_ = None
        return (x_25,)
