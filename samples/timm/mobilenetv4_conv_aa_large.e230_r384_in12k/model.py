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
        L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_head_buffers_running_mean_: torch.Tensor,
        L_self_modules_norm_head_buffers_running_var_: torch.Tensor,
        L_self_modules_norm_head_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_head_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_4_modules_0_modules_conv_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_0_modules_conv_parameters_weight_
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
        l_self_modules_conv_head_parameters_weight_ = (
            L_self_modules_conv_head_parameters_weight_
        )
        l_self_modules_norm_head_buffers_running_mean_ = (
            L_self_modules_norm_head_buffers_running_mean_
        )
        l_self_modules_norm_head_buffers_running_var_ = (
            L_self_modules_norm_head_buffers_running_var_
        )
        l_self_modules_norm_head_parameters_weight_ = (
            L_self_modules_norm_head_parameters_weight_
        )
        l_self_modules_norm_head_parameters_bias_ = (
            L_self_modules_norm_head_parameters_bias_
        )
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
        x_6 = torch._C._nn.avg_pool2d(x_5, 2, 2, 0, False, True, None)
        x_5 = None
        x_7 = torch.conv2d(
            x_6,
            l_self_modules_blocks_modules_0_modules_0_modules_conv_pwl_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_6 = l_self_modules_blocks_modules_0_modules_0_modules_conv_pwl_parameters_weight_ = (None)
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_
        ) = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_8 = l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_conv_parameters_weight_ = (None)
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_11 = torch.conv2d(
            x_10,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_10 = l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_12 = torch.nn.functional.batch_norm(
            x_11,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_11 = l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_13 = torch.nn.functional.relu(x_12, inplace=True)
        x_12 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        x_13 = l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_14 = l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_16 = torch.nn.functional.relu(x_15, inplace=True)
        x_15 = None
        x_17 = torch._C._nn.avg_pool2d(x_16, 2, 2, 0, False, True, None)
        x_16 = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_17 = l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_20 = torch.conv2d(
            x_19,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_21 = l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_24 = torch.nn.functional.relu(x_23, inplace=True)
        x_23 = None
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        x_24 = l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_27 = torch.nn.functional.relu(x_26, inplace=True)
        x_26 = None
        x_28 = torch.conv2d(
            x_27,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_27 = l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_30 = x_29 + x_19
        x_29 = x_19 = None
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        x_30 = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_conv_parameters_weight_ = (None)
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_33 = torch.conv2d(
            x_32,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_32 = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_35 = torch.nn.functional.relu(x_34, inplace=True)
        x_34 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            384,
        )
        x_35 = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_38 = torch.nn.functional.relu(x_37, inplace=True)
        x_37 = None
        x_39 = torch._C._nn.avg_pool2d(x_38, 2, 2, 0, False, True, None)
        x_38 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_43 = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_45 = torch.nn.functional.batch_norm(
            x_44,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_44 = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_46 = torch.nn.functional.relu(x_45, inplace=True)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_46 = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_52 = x_51 + x_41
        x_51 = x_41 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_55 = torch.conv2d(
            x_54,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_54 = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_55 = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_57 = torch.nn.functional.relu(x_56, inplace=True)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_57 = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_60 = torch.nn.functional.relu(x_59, inplace=True)
        x_59 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_60 = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_63 = x_62 + x_52
        x_62 = x_52 = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_68 = torch.nn.functional.relu(x_67, inplace=True)
        x_67 = None
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_68 = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_71 = torch.nn.functional.relu(x_70, inplace=True)
        x_70 = None
        x_72 = torch.conv2d(
            x_71,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_71 = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_74 = x_73 + x_63
        x_73 = x_63 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_77 = torch.conv2d(
            x_76,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_76 = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_77 = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_79 = torch.nn.functional.relu(x_78, inplace=True)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_79 = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_82 = torch.nn.functional.relu(x_81, inplace=True)
        x_81 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_85 = x_84 + x_74
        x_84 = x_74 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_88 = torch.conv2d(
            x_87,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_87 = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_90 = torch.nn.functional.relu(x_89, inplace=True)
        x_89 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_90 = l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_93 = torch.nn.functional.relu(x_92, inplace=True)
        x_92 = None
        x_94 = torch.conv2d(
            x_93,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_93 = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_96 = x_95 + x_85
        x_95 = x_85 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_97 = l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_99 = torch.conv2d(
            x_98,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_98 = l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_99 = l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_101 = torch.nn.functional.relu(x_100, inplace=True)
        x_100 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_101 = l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_102 = l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_104 = torch.nn.functional.relu(x_103, inplace=True)
        x_103 = None
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_104 = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_106 = torch.nn.functional.batch_norm(
            x_105,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_105 = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_107 = x_106 + x_96
        x_106 = x_96 = None
        x_108 = torch.conv2d(
            x_107,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_108 = l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_112 = torch.nn.functional.relu(x_111, inplace=True)
        x_111 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_112 = l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_115 = torch.nn.functional.relu(x_114, inplace=True)
        x_114 = None
        x_116 = torch.conv2d(
            x_115,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_115 = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_117 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_116 = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_118 = x_117 + x_107
        x_117 = x_107 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_121 = torch.conv2d(
            x_120,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_120 = l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_121 = l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_123 = torch.nn.functional.relu(x_122, inplace=True)
        x_122 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_123 = l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_124 = l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_126 = torch.nn.functional.relu(x_125, inplace=True)
        x_125 = None
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_126 = l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_127 = l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_129 = x_128 + x_118
        x_128 = x_118 = None
        x_130 = torch.conv2d(
            x_129,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_130 = l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_132 = torch.conv2d(
            x_131,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_131 = l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_133 = torch.nn.functional.batch_norm(
            x_132,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_132 = l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_134 = torch.nn.functional.relu(x_133, inplace=True)
        x_133 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_134 = l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_136 = torch.nn.functional.batch_norm(
            x_135,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_135 = l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_137 = torch.nn.functional.relu(x_136, inplace=True)
        x_136 = None
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_137 = l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_138 = l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_140 = x_139 + x_129
        x_139 = x_129 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_141 = l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_143 = torch.conv2d(
            x_142,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_142 = l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_144 = torch.nn.functional.batch_norm(
            x_143,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_143 = l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_145 = torch.nn.functional.relu(x_144, inplace=True)
        x_144 = None
        x_146 = torch.conv2d(
            x_145,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_145 = l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_147 = torch.nn.functional.batch_norm(
            x_146,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_146 = l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_148 = x_147 + x_140
        x_147 = x_140 = None
        x_149 = torch.conv2d(
            x_148,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        x_148 = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_conv_parameters_weight_ = (None)
        x_150 = torch.nn.functional.batch_norm(
            x_149,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_149 = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_151 = torch.conv2d(
            x_150,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_150 = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_151 = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_153 = torch.nn.functional.relu(x_152, inplace=True)
        x_152 = None
        x_154 = torch.conv2d(
            x_153,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_153 = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_156 = torch.nn.functional.relu(x_155, inplace=True)
        x_155 = None
        x_157 = torch._C._nn.avg_pool2d(x_156, 2, 2, 0, False, True, None)
        x_156 = None
        x_158 = torch.conv2d(
            x_157,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_157 = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_159 = torch.nn.functional.batch_norm(
            x_158,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_158 = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_160 = torch.conv2d(
            x_159,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_161 = torch.nn.functional.batch_norm(
            x_160,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_160 = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_161 = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_163 = torch.nn.functional.batch_norm(
            x_162,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_162 = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_164 = torch.nn.functional.relu(x_163, inplace=True)
        x_163 = None
        x_165 = torch.conv2d(
            x_164,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2048,
        )
        x_164 = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_166 = torch.nn.functional.batch_norm(
            x_165,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_165 = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_167 = torch.nn.functional.relu(x_166, inplace=True)
        x_166 = None
        x_168 = torch.conv2d(
            x_167,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_167 = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_169 = torch.nn.functional.batch_norm(
            x_168,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_168 = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_170 = x_169 + x_159
        x_169 = x_159 = None
        x_171 = torch.conv2d(
            x_170,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_172 = torch.nn.functional.batch_norm(
            x_171,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_171 = l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_173 = torch.conv2d(
            x_172,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_172 = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_174 = torch.nn.functional.batch_norm(
            x_173,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_173 = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_175 = torch.nn.functional.relu(x_174, inplace=True)
        x_174 = None
        x_176 = torch.conv2d(
            x_175,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2048,
        )
        x_175 = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_177 = torch.nn.functional.batch_norm(
            x_176,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_176 = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_178 = torch.nn.functional.relu(x_177, inplace=True)
        x_177 = None
        x_179 = torch.conv2d(
            x_178,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_178 = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_180 = torch.nn.functional.batch_norm(
            x_179,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_179 = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_181 = x_180 + x_170
        x_180 = x_170 = None
        x_182 = torch.conv2d(
            x_181,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_183 = torch.nn.functional.batch_norm(
            x_182,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_182 = l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_184 = torch.conv2d(
            x_183,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_183 = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_185 = torch.nn.functional.batch_norm(
            x_184,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_184 = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_186 = torch.nn.functional.relu(x_185, inplace=True)
        x_185 = None
        x_187 = torch.conv2d(
            x_186,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2048,
        )
        x_186 = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_188 = torch.nn.functional.batch_norm(
            x_187,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_187 = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_189 = torch.nn.functional.relu(x_188, inplace=True)
        x_188 = None
        x_190 = torch.conv2d(
            x_189,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_189 = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_191 = torch.nn.functional.batch_norm(
            x_190,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_190 = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_192 = x_191 + x_181
        x_191 = x_181 = None
        x_193 = torch.conv2d(
            x_192,
            l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_194 = torch.nn.functional.batch_norm(
            x_193,
            l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_193 = l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_195 = torch.conv2d(
            x_194,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_194 = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_196 = torch.nn.functional.batch_norm(
            x_195,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_195 = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_197 = torch.nn.functional.relu(x_196, inplace=True)
        x_196 = None
        x_198 = torch.conv2d(
            x_197,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_197 = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_199 = torch.nn.functional.batch_norm(
            x_198,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_198 = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_200 = x_199 + x_192
        x_199 = x_192 = None
        x_201 = torch.conv2d(
            x_200,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_202 = torch.nn.functional.batch_norm(
            x_201,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_201 = l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_203 = torch.conv2d(
            x_202,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_202 = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_204 = torch.nn.functional.batch_norm(
            x_203,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_203 = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_205 = torch.nn.functional.relu(x_204, inplace=True)
        x_204 = None
        x_206 = torch.conv2d(
            x_205,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_205 = l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_207 = torch.nn.functional.batch_norm(
            x_206,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_206 = l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_208 = torch.nn.functional.relu(x_207, inplace=True)
        x_207 = None
        x_209 = torch.conv2d(
            x_208,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_208 = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_210 = torch.nn.functional.batch_norm(
            x_209,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_209 = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_211 = x_210 + x_200
        x_210 = x_200 = None
        x_212 = torch.conv2d(
            x_211,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_213 = torch.nn.functional.batch_norm(
            x_212,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_212 = l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_214 = torch.conv2d(
            x_213,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_213 = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_215 = torch.nn.functional.batch_norm(
            x_214,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_214 = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_216 = torch.nn.functional.relu(x_215, inplace=True)
        x_215 = None
        x_217 = torch.conv2d(
            x_216,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_216 = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_218 = torch.nn.functional.batch_norm(
            x_217,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_217 = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_219 = x_218 + x_211
        x_218 = x_211 = None
        x_220 = torch.conv2d(
            x_219,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_221 = torch.nn.functional.batch_norm(
            x_220,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_220 = l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_222 = torch.conv2d(
            x_221,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_221 = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_223 = torch.nn.functional.batch_norm(
            x_222,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_222 = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_224 = torch.nn.functional.relu(x_223, inplace=True)
        x_223 = None
        x_225 = torch.conv2d(
            x_224,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_224 = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_226 = torch.nn.functional.batch_norm(
            x_225,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_225 = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_227 = x_226 + x_219
        x_226 = x_219 = None
        x_228 = torch.conv2d(
            x_227,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_229 = torch.nn.functional.batch_norm(
            x_228,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_228 = l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_230 = torch.conv2d(
            x_229,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_229 = l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_231 = torch.nn.functional.batch_norm(
            x_230,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_230 = l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_232 = torch.nn.functional.relu(x_231, inplace=True)
        x_231 = None
        x_233 = torch.conv2d(
            x_232,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_232 = l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_234 = torch.nn.functional.batch_norm(
            x_233,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_233 = l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_235 = torch.nn.functional.relu(x_234, inplace=True)
        x_234 = None
        x_236 = torch.conv2d(
            x_235,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_235 = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_237 = torch.nn.functional.batch_norm(
            x_236,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_236 = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_238 = x_237 + x_227
        x_237 = x_227 = None
        x_239 = torch.conv2d(
            x_238,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_240 = torch.nn.functional.batch_norm(
            x_239,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_239 = l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_241 = torch.conv2d(
            x_240,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_240 = l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_242 = torch.nn.functional.batch_norm(
            x_241,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_241 = l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_243 = torch.nn.functional.relu(x_242, inplace=True)
        x_242 = None
        x_244 = torch.conv2d(
            x_243,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2048,
        )
        x_243 = l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_245 = torch.nn.functional.batch_norm(
            x_244,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_244 = l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_246 = torch.nn.functional.relu(x_245, inplace=True)
        x_245 = None
        x_247 = torch.conv2d(
            x_246,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_246 = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_248 = torch.nn.functional.batch_norm(
            x_247,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_247 = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_249 = x_248 + x_238
        x_248 = x_238 = None
        x_250 = torch.conv2d(
            x_249,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_251 = torch.nn.functional.batch_norm(
            x_250,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_250 = l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_252 = torch.conv2d(
            x_251,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_251 = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_253 = torch.nn.functional.batch_norm(
            x_252,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_252 = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_254 = torch.nn.functional.relu(x_253, inplace=True)
        x_253 = None
        x_255 = torch.conv2d(
            x_254,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_254 = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_256 = torch.nn.functional.batch_norm(
            x_255,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_255 = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_257 = x_256 + x_249
        x_256 = x_249 = None
        x_258 = torch.conv2d(
            x_257,
            l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_259 = torch.nn.functional.batch_norm(
            x_258,
            l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_258 = l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_260 = torch.conv2d(
            x_259,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_259 = l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_261 = torch.nn.functional.batch_norm(
            x_260,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_260 = l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_262 = torch.nn.functional.relu(x_261, inplace=True)
        x_261 = None
        x_263 = torch.conv2d(
            x_262,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_262 = l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_264 = torch.nn.functional.batch_norm(
            x_263,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_263 = l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_265 = x_264 + x_257
        x_264 = x_257 = None
        x_266 = torch.conv2d(
            x_265,
            l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_267 = torch.nn.functional.batch_norm(
            x_266,
            l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_266 = l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_268 = torch.conv2d(
            x_267,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_267 = l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_269 = torch.nn.functional.batch_norm(
            x_268,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_268 = l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_270 = torch.nn.functional.relu(x_269, inplace=True)
        x_269 = None
        x_271 = torch.conv2d(
            x_270,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_270 = l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_272 = torch.nn.functional.batch_norm(
            x_271,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_271 = l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_273 = x_272 + x_265
        x_272 = x_265 = None
        x_274 = torch.conv2d(
            x_273,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_273 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_parameters_weight_
        ) = None
        x_275 = torch.nn.functional.batch_norm(
            x_274,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_274 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_276 = torch.nn.functional.relu(x_275, inplace=True)
        x_275 = None
        x_277 = torch.nn.functional.adaptive_avg_pool2d(x_276, 1)
        x_276 = None
        x_278 = torch.conv2d(
            x_277,
            l_self_modules_conv_head_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_277 = l_self_modules_conv_head_parameters_weight_ = None
        x_279 = torch.nn.functional.batch_norm(
            x_278,
            l_self_modules_norm_head_buffers_running_mean_,
            l_self_modules_norm_head_buffers_running_var_,
            l_self_modules_norm_head_parameters_weight_,
            l_self_modules_norm_head_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_278 = (
            l_self_modules_norm_head_buffers_running_mean_
        ) = (
            l_self_modules_norm_head_buffers_running_var_
        ) = (
            l_self_modules_norm_head_parameters_weight_
        ) = l_self_modules_norm_head_parameters_bias_ = None
        x_280 = torch.nn.functional.relu(x_279, inplace=True)
        x_279 = None
        x_281 = x_280.flatten(1, -1)
        x_280 = None
        x_282 = torch._C._nn.linear(
            x_281,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_281 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_282,)
