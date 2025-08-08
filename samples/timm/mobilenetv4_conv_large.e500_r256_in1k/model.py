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
            (2, 2),
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
            l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_7 = l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_conv_parameters_weight_ = (None)
        x_9 = torch.nn.functional.batch_norm(
            x_8,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_8 = l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_11 = torch.nn.functional.batch_norm(
            x_10,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_10 = l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_12 = torch.nn.functional.relu(x_11, inplace=True)
        x_11 = None
        x_13 = torch.conv2d(
            x_12,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            192,
        )
        x_12 = l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_14 = torch.nn.functional.batch_norm(
            x_13,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_13 = l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_15 = torch.nn.functional.relu(x_14, inplace=True)
        x_14 = None
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_15 = l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_18 = torch.conv2d(
            x_17,
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
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_1_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_20 = torch.conv2d(
            x_19,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_19 = l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_1_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_22 = torch.nn.functional.relu(x_21, inplace=True)
        x_21 = None
        x_23 = torch.conv2d(
            x_22,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        x_22 = l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_1_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_25 = torch.nn.functional.relu(x_24, inplace=True)
        x_24 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_1_modules_1_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_28 = x_27 + x_17
        x_27 = x_17 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            96,
        )
        x_28 = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_conv_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_30 = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_33 = torch.nn.functional.relu(x_32, inplace=True)
        x_32 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            384,
        )
        x_33 = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_36 = torch.nn.functional.relu(x_35, inplace=True)
        x_35 = None
        x_37 = torch.conv2d(
            x_36,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_36 = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_37 = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_39 = torch.conv2d(
            x_38,
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
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_39 = l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_41 = torch.conv2d(
            x_40,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_40 = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_42 = torch.nn.functional.batch_norm(
            x_41,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_41 = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_43 = torch.nn.functional.relu(x_42, inplace=True)
        x_42 = None
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_43 = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_45 = torch.nn.functional.batch_norm(
            x_44,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_44 = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_46 = torch.nn.functional.relu(x_45, inplace=True)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_1_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_49 = x_48 + x_38
        x_48 = x_38 = None
        x_50 = torch.conv2d(
            x_49,
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
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_51 = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_54 = torch.nn.functional.relu(x_53, inplace=True)
        x_53 = None
        x_55 = torch.conv2d(
            x_54,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_54 = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_55 = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_57 = torch.nn.functional.relu(x_56, inplace=True)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_2_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_60 = x_59 + x_49
        x_59 = x_49 = None
        x_61 = torch.conv2d(
            x_60,
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
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_63 = torch.conv2d(
            x_62,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_62 = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_65 = torch.nn.functional.relu(x_64, inplace=True)
        x_64 = None
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_65 = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_68 = torch.nn.functional.relu(x_67, inplace=True)
        x_67 = None
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_68 = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_3_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_71 = x_70 + x_60
        x_70 = x_60 = None
        x_72 = torch.conv2d(
            x_71,
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
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_74 = torch.conv2d(
            x_73,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_73 = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_75 = torch.nn.functional.batch_norm(
            x_74,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_74 = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_76 = torch.nn.functional.relu(x_75, inplace=True)
        x_75 = None
        x_77 = torch.conv2d(
            x_76,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            768,
        )
        x_76 = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_77 = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_4_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_79 = torch.nn.functional.relu(x_78, inplace=True)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_4_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_82 = x_81 + x_71
        x_81 = x_71 = None
        x_83 = torch.conv2d(
            x_82,
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
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_5_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_85 = torch.conv2d(
            x_84,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_84 = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_85 = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_87 = torch.nn.functional.relu(x_86, inplace=True)
        x_86 = None
        x_88 = torch.conv2d(
            x_87,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_87 = l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_5_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_90 = torch.nn.functional.relu(x_89, inplace=True)
        x_89 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_90 = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_93 = x_92 + x_82
        x_92 = x_82 = None
        x_94 = torch.conv2d(
            x_93,
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
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_6_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_96 = torch.conv2d(
            x_95,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_95 = l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_96 = l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_98 = torch.nn.functional.relu(x_97, inplace=True)
        x_97 = None
        x_99 = torch.conv2d(
            x_98,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_98 = l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_99 = l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_6_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_101 = torch.nn.functional.relu(x_100, inplace=True)
        x_100 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_101 = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_102 = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_104 = x_103 + x_93
        x_103 = x_93 = None
        x_105 = torch.conv2d(
            x_104,
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
        x_106 = torch.nn.functional.batch_norm(
            x_105,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_105 = l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_106 = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_109 = torch.nn.functional.relu(x_108, inplace=True)
        x_108 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_109 = l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_7_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_112 = torch.nn.functional.relu(x_111, inplace=True)
        x_111 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_112 = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_115 = x_114 + x_104
        x_114 = x_104 = None
        x_116 = torch.conv2d(
            x_115,
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
        x_117 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_116 = l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_8_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_119 = torch.nn.functional.batch_norm(
            x_118,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_118 = l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_8_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_120 = torch.nn.functional.relu(x_119, inplace=True)
        x_119 = None
        x_121 = torch.conv2d(
            x_120,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_120 = l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_121 = l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_8_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_123 = torch.nn.functional.relu(x_122, inplace=True)
        x_122 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_123 = l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_124 = l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_8_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_126 = x_125 + x_115
        x_125 = x_115 = None
        x_127 = torch.conv2d(
            x_126,
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
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_127 = l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_9_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_129 = torch.conv2d(
            x_128,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_128 = l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_130 = torch.nn.functional.batch_norm(
            x_129,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_129 = l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_9_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_131 = torch.nn.functional.relu(x_130, inplace=True)
        x_130 = None
        x_132 = torch.conv2d(
            x_131,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        x_131 = l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_133 = torch.nn.functional.batch_norm(
            x_132,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_132 = l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_9_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_134 = torch.nn.functional.relu(x_133, inplace=True)
        x_133 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_134 = l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_136 = torch.nn.functional.batch_norm(
            x_135,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_135 = l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_9_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_137 = x_136 + x_126
        x_136 = x_126 = None
        x_138 = torch.conv2d(
            x_137,
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
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_138 = l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_10_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_140 = torch.conv2d(
            x_139,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_141 = torch.nn.functional.batch_norm(
            x_140,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_140 = l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_10_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_142 = torch.nn.functional.relu(x_141, inplace=True)
        x_141 = None
        x_143 = torch.conv2d(
            x_142,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_142 = l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_144 = torch.nn.functional.batch_norm(
            x_143,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_143 = l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_10_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_145 = x_144 + x_137
        x_144 = x_137 = None
        x_146 = torch.conv2d(
            x_145,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            192,
        )
        x_145 = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_conv_parameters_weight_ = (None)
        x_147 = torch.nn.functional.batch_norm(
            x_146,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_146 = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_147 = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_150 = torch.nn.functional.relu(x_149, inplace=True)
        x_149 = None
        x_151 = torch.conv2d(
            x_150,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            768,
        )
        x_150 = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_151 = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_153 = torch.nn.functional.relu(x_152, inplace=True)
        x_152 = None
        x_154 = torch.conv2d(
            x_153,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_153 = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_156 = torch.conv2d(
            x_155,
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
        x_157 = torch.nn.functional.batch_norm(
            x_156,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_156 = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_158 = torch.conv2d(
            x_157,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_157 = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_159 = torch.nn.functional.batch_norm(
            x_158,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_158 = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_160 = torch.nn.functional.relu(x_159, inplace=True)
        x_159 = None
        x_161 = torch.conv2d(
            x_160,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2048,
        )
        x_160 = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_162 = torch.nn.functional.batch_norm(
            x_161,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_161 = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_163 = torch.nn.functional.relu(x_162, inplace=True)
        x_162 = None
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_163 = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_165 = torch.nn.functional.batch_norm(
            x_164,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_164 = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_166 = x_165 + x_155
        x_165 = x_155 = None
        x_167 = torch.conv2d(
            x_166,
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
        x_168 = torch.nn.functional.batch_norm(
            x_167,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_167 = l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_169 = torch.conv2d(
            x_168,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_168 = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_170 = torch.nn.functional.batch_norm(
            x_169,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_169 = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_171 = torch.nn.functional.relu(x_170, inplace=True)
        x_170 = None
        x_172 = torch.conv2d(
            x_171,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2048,
        )
        x_171 = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_173 = torch.nn.functional.batch_norm(
            x_172,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_172 = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_174 = torch.nn.functional.relu(x_173, inplace=True)
        x_173 = None
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_174 = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_176 = torch.nn.functional.batch_norm(
            x_175,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_175 = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_177 = x_176 + x_166
        x_176 = x_166 = None
        x_178 = torch.conv2d(
            x_177,
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
        x_179 = torch.nn.functional.batch_norm(
            x_178,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_178 = l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_180 = torch.conv2d(
            x_179,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_179 = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_181 = torch.nn.functional.batch_norm(
            x_180,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_180 = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_182 = torch.nn.functional.relu(x_181, inplace=True)
        x_181 = None
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2048,
        )
        x_182 = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_184 = torch.nn.functional.batch_norm(
            x_183,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_183 = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_185 = torch.nn.functional.relu(x_184, inplace=True)
        x_184 = None
        x_186 = torch.conv2d(
            x_185,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_185 = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_187 = torch.nn.functional.batch_norm(
            x_186,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_186 = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_188 = x_187 + x_177
        x_187 = x_177 = None
        x_189 = torch.conv2d(
            x_188,
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
        x_190 = torch.nn.functional.batch_norm(
            x_189,
            l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_189 = l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_4_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_191 = torch.conv2d(
            x_190,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_190 = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_192 = torch.nn.functional.batch_norm(
            x_191,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_191 = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_193 = torch.nn.functional.relu(x_192, inplace=True)
        x_192 = None
        x_194 = torch.conv2d(
            x_193,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_193 = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_195 = torch.nn.functional.batch_norm(
            x_194,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_194 = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_196 = x_195 + x_188
        x_195 = x_188 = None
        x_197 = torch.conv2d(
            x_196,
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
        x_198 = torch.nn.functional.batch_norm(
            x_197,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_197 = l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_199 = torch.conv2d(
            x_198,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_198 = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_200 = torch.nn.functional.batch_norm(
            x_199,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_199 = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_201 = torch.nn.functional.relu(x_200, inplace=True)
        x_200 = None
        x_202 = torch.conv2d(
            x_201,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_201 = l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_203 = torch.nn.functional.batch_norm(
            x_202,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_202 = l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_204 = torch.nn.functional.relu(x_203, inplace=True)
        x_203 = None
        x_205 = torch.conv2d(
            x_204,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_204 = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_206 = torch.nn.functional.batch_norm(
            x_205,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_205 = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_207 = x_206 + x_196
        x_206 = x_196 = None
        x_208 = torch.conv2d(
            x_207,
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
        x_209 = torch.nn.functional.batch_norm(
            x_208,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_208 = l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_210 = torch.conv2d(
            x_209,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_209 = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_211 = torch.nn.functional.batch_norm(
            x_210,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_210 = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_212 = torch.nn.functional.relu(x_211, inplace=True)
        x_211 = None
        x_213 = torch.conv2d(
            x_212,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_212 = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_214 = torch.nn.functional.batch_norm(
            x_213,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_213 = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_215 = x_214 + x_207
        x_214 = x_207 = None
        x_216 = torch.conv2d(
            x_215,
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
        x_217 = torch.nn.functional.batch_norm(
            x_216,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_216 = l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_218 = torch.conv2d(
            x_217,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_217 = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_219 = torch.nn.functional.batch_norm(
            x_218,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_218 = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_220 = torch.nn.functional.relu(x_219, inplace=True)
        x_219 = None
        x_221 = torch.conv2d(
            x_220,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_220 = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_222 = torch.nn.functional.batch_norm(
            x_221,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_221 = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_223 = x_222 + x_215
        x_222 = x_215 = None
        x_224 = torch.conv2d(
            x_223,
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
        x_225 = torch.nn.functional.batch_norm(
            x_224,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_224 = l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_8_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_226 = torch.conv2d(
            x_225,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_225 = l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_227 = torch.nn.functional.batch_norm(
            x_226,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_226 = l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_228 = torch.nn.functional.relu(x_227, inplace=True)
        x_227 = None
        x_229 = torch.conv2d(
            x_228,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2048,
        )
        x_228 = l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_230 = torch.nn.functional.batch_norm(
            x_229,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_229 = l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_8_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_231 = torch.nn.functional.relu(x_230, inplace=True)
        x_230 = None
        x_232 = torch.conv2d(
            x_231,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_231 = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_233 = torch.nn.functional.batch_norm(
            x_232,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_232 = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_234 = x_233 + x_223
        x_233 = x_223 = None
        x_235 = torch.conv2d(
            x_234,
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
        x_236 = torch.nn.functional.batch_norm(
            x_235,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_235 = l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_9_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_237 = torch.conv2d(
            x_236,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_236 = l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_238 = torch.nn.functional.batch_norm(
            x_237,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_237 = l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_239 = torch.nn.functional.relu(x_238, inplace=True)
        x_238 = None
        x_240 = torch.conv2d(
            x_239,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2048,
        )
        x_239 = l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_241 = torch.nn.functional.batch_norm(
            x_240,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_240 = l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_9_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_242 = torch.nn.functional.relu(x_241, inplace=True)
        x_241 = None
        x_243 = torch.conv2d(
            x_242,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_242 = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_244 = torch.nn.functional.batch_norm(
            x_243,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_243 = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_245 = x_244 + x_234
        x_244 = x_234 = None
        x_246 = torch.conv2d(
            x_245,
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
        x_247 = torch.nn.functional.batch_norm(
            x_246,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_246 = l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_248 = torch.conv2d(
            x_247,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_247 = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_249 = torch.nn.functional.batch_norm(
            x_248,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_248 = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_250 = torch.nn.functional.relu(x_249, inplace=True)
        x_249 = None
        x_251 = torch.conv2d(
            x_250,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_250 = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_252 = torch.nn.functional.batch_norm(
            x_251,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_251 = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_253 = x_252 + x_245
        x_252 = x_245 = None
        x_254 = torch.conv2d(
            x_253,
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
        x_255 = torch.nn.functional.batch_norm(
            x_254,
            l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_254 = l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_11_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_256 = torch.conv2d(
            x_255,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_255 = l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_257 = torch.nn.functional.batch_norm(
            x_256,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_256 = l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_11_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_258 = torch.nn.functional.relu(x_257, inplace=True)
        x_257 = None
        x_259 = torch.conv2d(
            x_258,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_258 = l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_260 = torch.nn.functional.batch_norm(
            x_259,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_259 = l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_11_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_261 = x_260 + x_253
        x_260 = x_253 = None
        x_262 = torch.conv2d(
            x_261,
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
        x_263 = torch.nn.functional.batch_norm(
            x_262,
            l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_262 = l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_12_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_264 = torch.conv2d(
            x_263,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_263 = l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_265 = torch.nn.functional.batch_norm(
            x_264,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_264 = l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_12_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_266 = torch.nn.functional.relu(x_265, inplace=True)
        x_265 = None
        x_267 = torch.conv2d(
            x_266,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_266 = l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_268 = torch.nn.functional.batch_norm(
            x_267,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_267 = l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_12_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_269 = x_268 + x_261
        x_268 = x_261 = None
        x_270 = torch.conv2d(
            x_269,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_269 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_parameters_weight_
        ) = None
        x_271 = torch.nn.functional.batch_norm(
            x_270,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_270 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_272 = torch.nn.functional.relu(x_271, inplace=True)
        x_271 = None
        x_273 = torch.nn.functional.adaptive_avg_pool2d(x_272, 1)
        x_272 = None
        x_274 = torch.conv2d(
            x_273,
            l_self_modules_conv_head_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_273 = l_self_modules_conv_head_parameters_weight_ = None
        x_275 = torch.nn.functional.batch_norm(
            x_274,
            l_self_modules_norm_head_buffers_running_mean_,
            l_self_modules_norm_head_buffers_running_var_,
            l_self_modules_norm_head_parameters_weight_,
            l_self_modules_norm_head_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_274 = (
            l_self_modules_norm_head_buffers_running_mean_
        ) = (
            l_self_modules_norm_head_buffers_running_var_
        ) = (
            l_self_modules_norm_head_parameters_weight_
        ) = l_self_modules_norm_head_parameters_bias_ = None
        x_276 = torch.nn.functional.relu(x_275, inplace=True)
        x_275 = None
        x_277 = x_276.flatten(1, -1)
        x_276 = None
        x_278 = torch._C._nn.linear(
            x_277,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_277 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_278,)
