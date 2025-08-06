import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_conv_stem_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_conv_stem_parameters_weight_ = (
            L_self_modules_conv_stem_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_bn1_buffers_running_mean_ = (
            L_self_modules_bn1_buffers_running_mean_
        )
        l_self_modules_bn1_buffers_running_var_ = (
            L_self_modules_bn1_buffers_running_var_
        )
        l_self_modules_bn1_parameters_weight_ = L_self_modules_bn1_parameters_weight_
        l_self_modules_bn1_parameters_bias_ = L_self_modules_bn1_parameters_bias_
        l_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_6_modules_0_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_conv_head_parameters_weight_ = (
            L_self_modules_conv_head_parameters_weight_
        )
        l_self_modules_bn2_buffers_running_mean_ = (
            L_self_modules_bn2_buffers_running_mean_
        )
        l_self_modules_bn2_buffers_running_var_ = (
            L_self_modules_bn2_buffers_running_var_
        )
        l_self_modules_bn2_parameters_weight_ = L_self_modules_bn2_parameters_weight_
        l_self_modules_bn2_parameters_bias_ = L_self_modules_bn2_parameters_bias_
        l_self_modules_classifier_parameters_weight_ = (
            L_self_modules_classifier_parameters_weight_
        )
        l_self_modules_classifier_parameters_bias_ = (
            L_self_modules_classifier_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_conv_stem_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_conv_stem_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_bn1_buffers_running_mean_,
            l_self_modules_bn1_buffers_running_var_,
            l_self_modules_bn1_parameters_weight_,
            l_self_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_bn1_parameters_weight_
        ) = l_self_modules_bn1_parameters_bias_ = None
        x_2 = torch.nn.functional.hardtanh(x_1, 0.0, 6.0, True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        x_2 = (
            l_self_modules_blocks_modules_0_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_5 = torch.nn.functional.hardtanh(x_4, 0.0, 6.0, True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = (
            l_self_modules_blocks_modules_0_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_8 = torch.conv2d(
            x_7,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_7 = (
            l_self_modules_blocks_modules_1_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_9 = torch.nn.functional.batch_norm(
            x_8,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_8 = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_10 = torch.nn.functional.hardtanh(x_9, 0.0, 6.0, True)
        x_9 = None
        x_11 = torch.conv2d(
            x_10,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            96,
        )
        x_10 = (
            l_self_modules_blocks_modules_1_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_12 = torch.nn.functional.batch_norm(
            x_11,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_11 = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_13 = torch.nn.functional.hardtanh(x_12, 0.0, 6.0, True)
        x_12 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_14 = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_1_modules_1_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_18 = torch.nn.functional.hardtanh(x_17, 0.0, 6.0, True)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            144,
        )
        x_18 = (
            l_self_modules_blocks_modules_1_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_21 = torch.nn.functional.hardtanh(x_20, 0.0, 6.0, True)
        x_20 = None
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_21 = l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_24 = x_23 + x_15
        x_23 = x_15 = None
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_blocks_modules_1_modules_2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_1_modules_2_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_27 = torch.nn.functional.hardtanh(x_26, 0.0, 6.0, True)
        x_26 = None
        x_28 = torch.conv2d(
            x_27,
            l_self_modules_blocks_modules_1_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            144,
        )
        x_27 = (
            l_self_modules_blocks_modules_1_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_30 = torch.nn.functional.hardtanh(x_29, 0.0, 6.0, True)
        x_29 = None
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_30 = l_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_33 = x_32 + x_24
        x_32 = x_24 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_33 = (
            l_self_modules_blocks_modules_2_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_36 = torch.nn.functional.hardtanh(x_35, 0.0, 6.0, True)
        x_35 = None
        x_37 = torch.conv2d(
            x_36,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            144,
        )
        x_36 = (
            l_self_modules_blocks_modules_2_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_37 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_39 = torch.nn.functional.hardtanh(x_38, 0.0, 6.0, True)
        x_38 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_1_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_44 = torch.nn.functional.hardtanh(x_43, 0.0, 6.0, True)
        x_43 = None
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        x_44 = (
            l_self_modules_blocks_modules_2_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_47 = torch.nn.functional.hardtanh(x_46, 0.0, 6.0, True)
        x_46 = None
        x_48 = torch.conv2d(
            x_47,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_47 = l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_48 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_50 = x_49 + x_41
        x_49 = x_41 = None
        x_51 = torch.conv2d(
            x_50,
            l_self_modules_blocks_modules_2_modules_2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_2_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_51 = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_53 = torch.nn.functional.hardtanh(x_52, 0.0, 6.0, True)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_blocks_modules_2_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        x_53 = (
            l_self_modules_blocks_modules_2_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_56 = torch.nn.functional.hardtanh(x_55, 0.0, 6.0, True)
        x_55 = None
        x_57 = torch.conv2d(
            x_56,
            l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_56 = l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_58 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_57 = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_59 = x_58 + x_50
        x_58 = x_50 = None
        x_60 = torch.conv2d(
            x_59,
            l_self_modules_blocks_modules_2_modules_3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_3_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_60 = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_62 = torch.nn.functional.hardtanh(x_61, 0.0, 6.0, True)
        x_61 = None
        x_63 = torch.conv2d(
            x_62,
            l_self_modules_blocks_modules_2_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        x_62 = (
            l_self_modules_blocks_modules_2_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_65 = torch.nn.functional.hardtanh(x_64, 0.0, 6.0, True)
        x_64 = None
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_68 = x_67 + x_59
        x_67 = x_59 = None
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_68 = (
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_71 = torch.nn.functional.hardtanh(x_70, 0.0, 6.0, True)
        x_70 = None
        x_72 = torch.conv2d(
            x_71,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            192,
        )
        x_71 = (
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_74 = torch.nn.functional.hardtanh(x_73, 0.0, 6.0, True)
        x_73 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_77 = torch.conv2d(
            x_76,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_77 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_79 = torch.nn.functional.hardtanh(x_78, 0.0, 6.0, True)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            432,
        )
        x_79 = (
            l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_82 = torch.nn.functional.hardtanh(x_81, 0.0, 6.0, True)
        x_81 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_85 = x_84 + x_76
        x_84 = x_76 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_88 = torch.nn.functional.hardtanh(x_87, 0.0, 6.0, True)
        x_87 = None
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            432,
        )
        x_88 = (
            l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_89 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_91 = torch.nn.functional.hardtanh(x_90, 0.0, 6.0, True)
        x_90 = None
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_91 = l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_94 = x_93 + x_85
        x_93 = x_85 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_97 = torch.nn.functional.hardtanh(x_96, 0.0, 6.0, True)
        x_96 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            432,
        )
        x_97 = (
            l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_98 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_100 = torch.nn.functional.hardtanh(x_99, 0.0, 6.0, True)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_100 = l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_103 = x_102 + x_94
        x_102 = x_94 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_blocks_modules_3_modules_4_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_4_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_104 = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_106 = torch.nn.functional.hardtanh(x_105, 0.0, 6.0, True)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            432,
        )
        x_106 = (
            l_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_
        ) = None
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_109 = torch.nn.functional.hardtanh(x_108, 0.0, 6.0, True)
        x_108 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_
        ) = None
        x_112 = x_111 + x_103
        x_111 = x_103 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_112 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_115 = torch.nn.functional.hardtanh(x_114, 0.0, 6.0, True)
        x_114 = None
        x_116 = torch.conv2d(
            x_115,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            432,
        )
        x_115 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_117 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_116 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_118 = torch.nn.functional.hardtanh(x_117, 0.0, 6.0, True)
        x_117 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_121 = torch.conv2d(
            x_120,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_121 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_123 = torch.nn.functional.hardtanh(x_122, 0.0, 6.0, True)
        x_122 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            624,
        )
        x_123 = (
            l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_124 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_126 = torch.nn.functional.hardtanh(x_125, 0.0, 6.0, True)
        x_125 = None
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_126 = l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_127 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_129 = x_128 + x_120
        x_128 = x_120 = None
        x_130 = torch.conv2d(
            x_129,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_2_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_130 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_132 = torch.nn.functional.hardtanh(x_131, 0.0, 6.0, True)
        x_131 = None
        x_133 = torch.conv2d(
            x_132,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            624,
        )
        x_132 = (
            l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_134 = torch.nn.functional.batch_norm(
            x_133,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_133 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_135 = torch.nn.functional.hardtanh(x_134, 0.0, 6.0, True)
        x_134 = None
        x_136 = torch.conv2d(
            x_135,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_135 = l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_138 = x_137 + x_129
        x_137 = x_129 = None
        x_139 = torch.conv2d(
            x_138,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_3_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_139 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_141 = torch.nn.functional.hardtanh(x_140, 0.0, 6.0, True)
        x_140 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            624,
        )
        x_141 = (
            l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_142 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_144 = torch.nn.functional.hardtanh(x_143, 0.0, 6.0, True)
        x_143 = None
        x_145 = torch.conv2d(
            x_144,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_144 = l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_146 = torch.nn.functional.batch_norm(
            x_145,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_145 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_147 = x_146 + x_138
        x_146 = x_138 = None
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_147 = (
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_150 = torch.nn.functional.hardtanh(x_149, 0.0, 6.0, True)
        x_149 = None
        x_151 = torch.conv2d(
            x_150,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            624,
        )
        x_150 = (
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_151 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_153 = torch.nn.functional.hardtanh(x_152, 0.0, 6.0, True)
        x_152 = None
        x_154 = torch.conv2d(
            x_153,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_153 = l_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_156 = torch.conv2d(
            x_155,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_157 = torch.nn.functional.batch_norm(
            x_156,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_156 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_158 = torch.nn.functional.hardtanh(x_157, 0.0, 6.0, True)
        x_157 = None
        x_159 = torch.conv2d(
            x_158,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1056,
        )
        x_158 = (
            l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_160 = torch.nn.functional.batch_norm(
            x_159,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_159 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_161 = torch.nn.functional.hardtanh(x_160, 0.0, 6.0, True)
        x_160 = None
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_161 = l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_163 = torch.nn.functional.batch_norm(
            x_162,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_162 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_164 = x_163 + x_155
        x_163 = x_155 = None
        x_165 = torch.conv2d(
            x_164,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_166 = torch.nn.functional.batch_norm(
            x_165,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_165 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_167 = torch.nn.functional.hardtanh(x_166, 0.0, 6.0, True)
        x_166 = None
        x_168 = torch.conv2d(
            x_167,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1056,
        )
        x_167 = (
            l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_169 = torch.nn.functional.batch_norm(
            x_168,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_168 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_170 = torch.nn.functional.hardtanh(x_169, 0.0, 6.0, True)
        x_169 = None
        x_171 = torch.conv2d(
            x_170,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_170 = l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_172 = torch.nn.functional.batch_norm(
            x_171,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_171 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_173 = x_172 + x_164
        x_172 = x_164 = None
        x_174 = torch.conv2d(
            x_173,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_175 = torch.nn.functional.batch_norm(
            x_174,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_174 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_176 = torch.nn.functional.hardtanh(x_175, 0.0, 6.0, True)
        x_175 = None
        x_177 = torch.conv2d(
            x_176,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1056,
        )
        x_176 = (
            l_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_178 = torch.nn.functional.batch_norm(
            x_177,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_177 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_179 = torch.nn.functional.hardtanh(x_178, 0.0, 6.0, True)
        x_178 = None
        x_180 = torch.conv2d(
            x_179,
            l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_179 = l_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_181 = torch.nn.functional.batch_norm(
            x_180,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_180 = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_182 = x_181 + x_173
        x_181 = x_173 = None
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_blocks_modules_6_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_182 = (
            l_self_modules_blocks_modules_6_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_184 = torch.nn.functional.batch_norm(
            x_183,
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_183 = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_185 = torch.nn.functional.hardtanh(x_184, 0.0, 6.0, True)
        x_184 = None
        x_186 = torch.conv2d(
            x_185,
            l_self_modules_blocks_modules_6_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1056,
        )
        x_185 = (
            l_self_modules_blocks_modules_6_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_187 = torch.nn.functional.batch_norm(
            x_186,
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_186 = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_188 = torch.nn.functional.hardtanh(x_187, 0.0, 6.0, True)
        x_187 = None
        x_189 = torch.conv2d(
            x_188,
            l_self_modules_blocks_modules_6_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_188 = l_self_modules_blocks_modules_6_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_190 = torch.nn.functional.batch_norm(
            x_189,
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_189 = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_191 = torch.conv2d(
            x_190,
            l_self_modules_conv_head_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_190 = l_self_modules_conv_head_parameters_weight_ = None
        x_192 = torch.nn.functional.batch_norm(
            x_191,
            l_self_modules_bn2_buffers_running_mean_,
            l_self_modules_bn2_buffers_running_var_,
            l_self_modules_bn2_parameters_weight_,
            l_self_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_191 = (
            l_self_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_bn2_parameters_weight_
        ) = l_self_modules_bn2_parameters_bias_ = None
        x_193 = torch.nn.functional.hardtanh(x_192, 0.0, 6.0, True)
        x_192 = None
        x_194 = torch.nn.functional.adaptive_avg_pool2d(x_193, 1)
        x_193 = None
        x_195 = x_194.flatten(1, -1)
        x_194 = None
        x_196 = torch._C._nn.linear(
            x_195,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_195 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_196,)
