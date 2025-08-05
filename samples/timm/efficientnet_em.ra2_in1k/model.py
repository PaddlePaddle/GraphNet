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
        L_self_modules_blocks_modules_0_modules_0_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_conv_exp_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_3_modules_5_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_4_modules_4_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_blocks_modules_0_modules_0_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_0_modules_0_modules_conv_exp_parameters_weight_
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
        l_self_modules_blocks_modules_0_modules_0_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_0_modules_0_modules_conv_pwl_parameters_weight_
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
        l_self_modules_blocks_modules_0_modules_1_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_0_modules_1_modules_conv_exp_parameters_weight_
        l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_1_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_0_modules_1_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_
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
        l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_
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
        l_self_modules_blocks_modules_1_modules_1_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_conv_exp_parameters_weight_
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
        l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_
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
        l_self_modules_blocks_modules_1_modules_2_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_1_modules_2_modules_conv_exp_parameters_weight_
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
        l_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_1_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_conv_exp_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_2_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_conv_exp_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_3_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_conv_exp_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_
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
        l_self_modules_blocks_modules_2_modules_4_modules_conv_exp_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_conv_exp_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_bias_
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
        l_self_modules_blocks_modules_3_modules_5_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_
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
        l_self_modules_blocks_modules_4_modules_4_modules_conv_pw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_conv_pw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_ = L_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_
        l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_
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
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_blocks_modules_0_modules_0_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_blocks_modules_0_modules_0_modules_conv_exp_parameters_weight_ = (None)
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
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_blocks_modules_0_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_blocks_modules_0_modules_0_modules_conv_pwl_parameters_weight_ = (None)
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
            l_self_modules_blocks_modules_0_modules_1_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_7 = l_self_modules_blocks_modules_0_modules_1_modules_conv_exp_parameters_weight_ = (None)
        x_9 = torch.nn.functional.batch_norm(
            x_8,
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_8 = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_10 = torch.nn.functional.relu(x_9, inplace=True)
        x_9 = None
        x_11 = torch.conv2d(
            x_10,
            l_self_modules_blocks_modules_0_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_10 = l_self_modules_blocks_modules_0_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_12 = torch.nn.functional.batch_norm(
            x_11,
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_11 = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_13 = torch.conv2d(
            x_12,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_12 = l_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_ = (None)
        x_14 = torch.nn.functional.batch_norm(
            x_13,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_13 = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_15 = torch.nn.functional.relu(x_14, inplace=True)
        x_14 = None
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_15 = l_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_1_modules_1_modules_conv_exp_parameters_weight_ = (
            None
        )
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_20 = torch.nn.functional.relu(x_19, inplace=True)
        x_19 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_20 = l_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_21 = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_23 = x_22 + x_17
        x_22 = x_17 = None
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_blocks_modules_1_modules_2_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_1_modules_2_modules_conv_exp_parameters_weight_ = (
            None
        )
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_24 = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_26 = torch.nn.functional.relu(x_25, inplace=True)
        x_25 = None
        x_27 = torch.conv2d(
            x_26,
            l_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_26 = l_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_28 = torch.nn.functional.batch_norm(
            x_27,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_27 = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_29 = x_28 + x_23
        x_28 = x_23 = None
        x_30 = torch.conv2d(
            x_29,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_29 = l_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_ = (None)
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_30 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_32 = torch.nn.functional.relu(x_31, inplace=True)
        x_31 = None
        x_33 = torch.conv2d(
            x_32,
            l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_32 = l_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_1_modules_conv_exp_parameters_weight_ = (
            None
        )
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_37 = torch.nn.functional.relu(x_36, inplace=True)
        x_36 = None
        x_38 = torch.conv2d(
            x_37,
            l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_40 = x_39 + x_34
        x_39 = x_34 = None
        x_41 = torch.conv2d(
            x_40,
            l_self_modules_blocks_modules_2_modules_2_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_2_modules_conv_exp_parameters_weight_ = (
            None
        )
        x_42 = torch.nn.functional.batch_norm(
            x_41,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_41 = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_43 = torch.nn.functional.relu(x_42, inplace=True)
        x_42 = None
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_43 = l_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_45 = torch.nn.functional.batch_norm(
            x_44,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_44 = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_46 = x_45 + x_40
        x_45 = x_40 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_blocks_modules_2_modules_3_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_3_modules_conv_exp_parameters_weight_ = (
            None
        )
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_52 = x_51 + x_46
        x_51 = x_46 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_blocks_modules_2_modules_4_modules_conv_exp_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_4_modules_conv_exp_parameters_weight_ = (
            None
        )
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_55 = torch.nn.functional.relu(x_54, inplace=True)
        x_54 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_55 = l_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_58 = x_57 + x_52
        x_57 = x_52 = None
        x_59 = torch.conv2d(
            x_58,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_58 = (
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_59 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_61 = torch.nn.functional.relu(x_60, inplace=True)
        x_60 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            384,
        )
        x_61 = (
            l_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_62 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_64 = torch.nn.functional.relu(x_63, inplace=True)
        x_63 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = l_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_65 = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_67 = torch.conv2d(
            x_66,
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
        x_68 = torch.nn.functional.batch_norm(
            x_67,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_67 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_69 = torch.nn.functional.relu(x_68, inplace=True)
        x_68 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_69 = (
            l_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_72 = torch.nn.functional.relu(x_71, inplace=True)
        x_71 = None
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_72 = l_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_75 = x_74 + x_66
        x_74 = x_66 = None
        x_76 = torch.conv2d(
            x_75,
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
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_78 = torch.nn.functional.relu(x_77, inplace=True)
        x_77 = None
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_78 = (
            l_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_81 = torch.nn.functional.relu(x_80, inplace=True)
        x_80 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_81 = l_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_84 = x_83 + x_75
        x_83 = x_75 = None
        x_85 = torch.conv2d(
            x_84,
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
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_85 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_87 = torch.nn.functional.relu(x_86, inplace=True)
        x_86 = None
        x_88 = torch.conv2d(
            x_87,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_87 = (
            l_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_90 = torch.nn.functional.relu(x_89, inplace=True)
        x_89 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_90 = l_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_93 = x_92 + x_84
        x_92 = x_84 = None
        x_94 = torch.conv2d(
            x_93,
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
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_96 = torch.nn.functional.relu(x_95, inplace=True)
        x_95 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_96 = (
            l_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_
        ) = None
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_97 = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_99 = torch.nn.functional.relu(x_98, inplace=True)
        x_98 = None
        x_100 = torch.conv2d(
            x_99,
            l_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_99 = l_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_100 = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_
        ) = None
        x_102 = x_101 + x_93
        x_101 = x_93 = None
        x_103 = torch.conv2d(
            x_102,
            l_self_modules_blocks_modules_3_modules_5_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_5_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_104 = torch.nn.functional.batch_norm(
            x_103,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_103 = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_
        ) = None
        x_105 = torch.nn.functional.relu(x_104, inplace=True)
        x_104 = None
        x_106 = torch.conv2d(
            x_105,
            l_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_105 = (
            l_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_
        ) = None
        x_107 = torch.nn.functional.batch_norm(
            x_106,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_106 = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_
        ) = None
        x_108 = torch.nn.functional.relu(x_107, inplace=True)
        x_107 = None
        x_109 = torch.conv2d(
            x_108,
            l_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_108 = l_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_ = (None)
        x_110 = torch.nn.functional.batch_norm(
            x_109,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_109 = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_
        ) = None
        x_111 = x_110 + x_102
        x_110 = x_102 = None
        x_112 = torch.conv2d(
            x_111,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_111 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_112 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_114 = torch.nn.functional.relu(x_113, inplace=True)
        x_113 = None
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_114 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_115 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_117 = torch.nn.functional.relu(x_116, inplace=True)
        x_116 = None
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_119 = torch.nn.functional.batch_norm(
            x_118,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_118 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_120 = torch.conv2d(
            x_119,
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
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_122 = torch.nn.functional.relu(x_121, inplace=True)
        x_121 = None
        x_123 = torch.conv2d(
            x_122,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1152,
        )
        x_122 = (
            l_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_125 = torch.nn.functional.relu(x_124, inplace=True)
        x_124 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_128 = x_127 + x_119
        x_127 = x_119 = None
        x_129 = torch.conv2d(
            x_128,
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
        x_130 = torch.nn.functional.batch_norm(
            x_129,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_129 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_131 = torch.nn.functional.relu(x_130, inplace=True)
        x_130 = None
        x_132 = torch.conv2d(
            x_131,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1152,
        )
        x_131 = (
            l_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_133 = torch.nn.functional.batch_norm(
            x_132,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_132 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_134 = torch.nn.functional.relu(x_133, inplace=True)
        x_133 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_134 = l_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_136 = torch.nn.functional.batch_norm(
            x_135,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_135 = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_137 = x_136 + x_128
        x_136 = x_128 = None
        x_138 = torch.conv2d(
            x_137,
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
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_138 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_
        ) = None
        x_140 = torch.nn.functional.relu(x_139, inplace=True)
        x_139 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1152,
        )
        x_140 = (
            l_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_
        ) = None
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_141 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_
        ) = None
        x_143 = torch.nn.functional.relu(x_142, inplace=True)
        x_142 = None
        x_144 = torch.conv2d(
            x_143,
            l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_143 = l_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_ = (None)
        x_145 = torch.nn.functional.batch_norm(
            x_144,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_144 = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_
        ) = None
        x_146 = x_145 + x_137
        x_145 = x_137 = None
        x_147 = torch.conv2d(
            x_146,
            l_self_modules_blocks_modules_4_modules_4_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_4_modules_4_modules_conv_pw_parameters_weight_ = (
            None
        )
        x_148 = torch.nn.functional.batch_norm(
            x_147,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_147 = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_
        ) = None
        x_149 = torch.nn.functional.relu(x_148, inplace=True)
        x_148 = None
        x_150 = torch.conv2d(
            x_149,
            l_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1152,
        )
        x_149 = (
            l_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_
        ) = None
        x_151 = torch.nn.functional.batch_norm(
            x_150,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_150 = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_
        ) = None
        x_152 = torch.nn.functional.relu(x_151, inplace=True)
        x_151 = None
        x_153 = torch.conv2d(
            x_152,
            l_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_152 = l_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_ = (None)
        x_154 = torch.nn.functional.batch_norm(
            x_153,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_153 = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_
        ) = None
        x_155 = x_154 + x_146
        x_154 = x_146 = None
        x_156 = torch.conv2d(
            x_155,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_155 = (
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_
        ) = None
        x_157 = torch.nn.functional.batch_norm(
            x_156,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_156 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_158 = torch.nn.functional.relu(x_157, inplace=True)
        x_157 = None
        x_159 = torch.conv2d(
            x_158,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            1152,
        )
        x_158 = (
            l_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_
        ) = None
        x_160 = torch.nn.functional.batch_norm(
            x_159,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_159 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_161 = torch.nn.functional.relu(x_160, inplace=True)
        x_160 = None
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_161 = l_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_163 = torch.nn.functional.batch_norm(
            x_162,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_162 = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_
        ) = None
        x_164 = torch.conv2d(
            x_163,
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
        x_165 = torch.nn.functional.batch_norm(
            x_164,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_164 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_
        ) = None
        x_166 = torch.nn.functional.relu(x_165, inplace=True)
        x_165 = None
        x_167 = torch.conv2d(
            x_166,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1536,
        )
        x_166 = (
            l_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_
        ) = None
        x_168 = torch.nn.functional.batch_norm(
            x_167,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_167 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_
        ) = None
        x_169 = torch.nn.functional.relu(x_168, inplace=True)
        x_168 = None
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_169 = l_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_ = (None)
        x_171 = torch.nn.functional.batch_norm(
            x_170,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_170 = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_
        ) = None
        x_172 = x_171 + x_163
        x_171 = x_163 = None
        x_173 = torch.conv2d(
            x_172,
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
        x_174 = torch.nn.functional.batch_norm(
            x_173,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_173 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_
        ) = None
        x_175 = torch.nn.functional.relu(x_174, inplace=True)
        x_174 = None
        x_176 = torch.conv2d(
            x_175,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1536,
        )
        x_175 = (
            l_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_
        ) = None
        x_177 = torch.nn.functional.batch_norm(
            x_176,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_176 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_
        ) = None
        x_178 = torch.nn.functional.relu(x_177, inplace=True)
        x_177 = None
        x_179 = torch.conv2d(
            x_178,
            l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_178 = l_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_ = (None)
        x_180 = torch.nn.functional.batch_norm(
            x_179,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_179 = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_
        ) = None
        x_181 = x_180 + x_172
        x_180 = x_172 = None
        x_182 = torch.conv2d(
            x_181,
            l_self_modules_conv_head_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_181 = l_self_modules_conv_head_parameters_weight_ = None
        x_183 = torch.nn.functional.batch_norm(
            x_182,
            l_self_modules_bn2_buffers_running_mean_,
            l_self_modules_bn2_buffers_running_var_,
            l_self_modules_bn2_parameters_weight_,
            l_self_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_182 = (
            l_self_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_bn2_parameters_weight_
        ) = l_self_modules_bn2_parameters_bias_ = None
        x_184 = torch.nn.functional.relu(x_183, inplace=True)
        x_183 = None
        x_185 = torch.nn.functional.adaptive_avg_pool2d(x_184, 1)
        x_184 = None
        x_186 = x_185.flatten(1, -1)
        x_185 = None
        x_187 = torch._C._nn.linear(
            x_186,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_186 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_187,)
