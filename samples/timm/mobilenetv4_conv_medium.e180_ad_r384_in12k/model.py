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
        L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_bias_
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
        l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_bias_
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
        l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_bias_
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
        l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_bias_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_conv_parameters_weight_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_conv_parameters_weight_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_weight_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_weight_
        l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_bias_ = L_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_bias_
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
            80,
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
            160,
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
            80,
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
            480,
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
            160,
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
            640,
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
            160,
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
            640,
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
            160,
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
            (2, 2),
            (1, 1),
            640,
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
            160,
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
            (1, 1),
            (1, 1),
            640,
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
            (1, 1),
            (1, 1),
            160,
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
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_87 = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_5_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_90 = x_89 + x_82
        x_89 = x_82 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_93 = torch.nn.functional.relu(x_92, inplace=True)
        x_92 = None
        x_94 = torch.conv2d(
            x_93,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_93 = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_6_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_96 = x_95 + x_90
        x_95 = x_90 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            160,
        )
        l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_97 = l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_7_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_99 = torch.conv2d(
            x_98,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_98 = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_99 = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_101 = torch.nn.functional.relu(x_100, inplace=True)
        x_100 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_101 = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_102 = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_2_modules_7_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_104 = x_103 + x_96
        x_103 = x_96 = None
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            160,
        )
        x_104 = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_conv_parameters_weight_ = (None)
        x_106 = torch.nn.functional.batch_norm(
            x_105,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_105 = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_106 = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_109 = torch.nn.functional.relu(x_108, inplace=True)
        x_108 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            960,
        )
        x_109 = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_112 = torch.nn.functional.relu(x_111, inplace=True)
        x_111 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_112 = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_0_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            256,
        )
        l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_115 = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_116 = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_117 = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_119 = torch.nn.functional.relu(x_118, inplace=True)
        x_118 = None
        x_120 = torch.conv2d(
            x_119,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1024,
        )
        x_119 = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_122 = torch.nn.functional.relu(x_121, inplace=True)
        x_121 = None
        x_123 = torch.conv2d(
            x_122,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_122 = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_1_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_125 = x_124 + x_114
        x_124 = x_114 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_128 = torch.conv2d(
            x_127,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_127 = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_130 = torch.nn.functional.relu(x_129, inplace=True)
        x_129 = None
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1024,
        )
        x_130 = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_133 = torch.nn.functional.relu(x_132, inplace=True)
        x_132 = None
        x_134 = torch.conv2d(
            x_133,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_133 = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_2_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_136 = x_135 + x_125
        x_135 = x_125 = None
        x_137 = torch.conv2d(
            x_136,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_137 = l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_139 = torch.conv2d(
            x_138,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_138 = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_139 = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_141 = torch.nn.functional.relu(x_140, inplace=True)
        x_140 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1024,
        )
        x_141 = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_142 = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_144 = torch.nn.functional.relu(x_143, inplace=True)
        x_143 = None
        x_145 = torch.conv2d(
            x_144,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_144 = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_146 = torch.nn.functional.batch_norm(
            x_145,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_145 = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_3_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_147 = x_146 + x_136
        x_146 = x_136 = None
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_150 = torch.nn.functional.relu(x_149, inplace=True)
        x_149 = None
        x_151 = torch.conv2d(
            x_150,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_150 = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_151 = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_4_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_153 = x_152 + x_147
        x_152 = x_147 = None
        x_154 = torch.conv2d(
            x_153,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_156 = torch.conv2d(
            x_155,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_155 = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_157 = torch.nn.functional.batch_norm(
            x_156,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_156 = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_158 = torch.nn.functional.relu(x_157, inplace=True)
        x_157 = None
        x_159 = torch.conv2d(
            x_158,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_158 = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_160 = torch.nn.functional.batch_norm(
            x_159,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_159 = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_5_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_161 = x_160 + x_153
        x_160 = x_153 = None
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_163 = torch.nn.functional.batch_norm(
            x_162,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_162 = l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_6_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_163 = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_165 = torch.nn.functional.batch_norm(
            x_164,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_164 = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_166 = torch.nn.functional.relu(x_165, inplace=True)
        x_165 = None
        x_167 = torch.conv2d(
            x_166,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            512,
        )
        x_166 = l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_168 = torch.nn.functional.batch_norm(
            x_167,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_167 = l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_6_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_169 = torch.nn.functional.relu(x_168, inplace=True)
        x_168 = None
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_169 = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_171 = torch.nn.functional.batch_norm(
            x_170,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_170 = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_6_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_172 = x_171 + x_161
        x_171 = x_161 = None
        x_173 = torch.conv2d(
            x_172,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            256,
        )
        l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_174 = torch.nn.functional.batch_norm(
            x_173,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_173 = l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_7_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_174 = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_176 = torch.nn.functional.batch_norm(
            x_175,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_175 = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_177 = torch.nn.functional.relu(x_176, inplace=True)
        x_176 = None
        x_178 = torch.conv2d(
            x_177,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1024,
        )
        x_177 = l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_conv_parameters_weight_ = (None)
        x_179 = torch.nn.functional.batch_norm(
            x_178,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_178 = l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_7_modules_dw_mid_modules_bn_parameters_bias_ = (None)
        x_180 = torch.nn.functional.relu(x_179, inplace=True)
        x_179 = None
        x_181 = torch.conv2d(
            x_180,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_180 = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_182 = torch.nn.functional.batch_norm(
            x_181,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_181 = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_7_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_183 = x_182 + x_172
        x_182 = x_172 = None
        x_184 = torch.conv2d(
            x_183,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_185 = torch.nn.functional.batch_norm(
            x_184,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_184 = l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_186 = torch.nn.functional.relu(x_185, inplace=True)
        x_185 = None
        x_187 = torch.conv2d(
            x_186,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_186 = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_188 = torch.nn.functional.batch_norm(
            x_187,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_187 = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_8_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_189 = x_188 + x_183
        x_188 = x_183 = None
        x_190 = torch.conv2d(
            x_189,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_191 = torch.nn.functional.batch_norm(
            x_190,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_190 = l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_192 = torch.nn.functional.relu(x_191, inplace=True)
        x_191 = None
        x_193 = torch.conv2d(
            x_192,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_192 = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_194 = torch.nn.functional.batch_norm(
            x_193,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_193 = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_9_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_195 = x_194 + x_189
        x_194 = x_189 = None
        x_196 = torch.conv2d(
            x_195,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            256,
        )
        l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_conv_parameters_weight_ = (
            None
        )
        x_197 = torch.nn.functional.batch_norm(
            x_196,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_196 = l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_10_modules_dw_start_modules_bn_parameters_bias_ = (None)
        x_198 = torch.conv2d(
            x_197,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_197 = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_conv_parameters_weight_ = (None)
        x_199 = torch.nn.functional.batch_norm(
            x_198,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_198 = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_exp_modules_bn_parameters_bias_ = (None)
        x_200 = torch.nn.functional.relu(x_199, inplace=True)
        x_199 = None
        x_201 = torch.conv2d(
            x_200,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_200 = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_conv_parameters_weight_ = (None)
        x_202 = torch.nn.functional.batch_norm(
            x_201,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_201 = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_weight_ = l_self_modules_blocks_modules_3_modules_10_modules_pw_proj_modules_bn_parameters_bias_ = (None)
        x_203 = x_202 + x_195
        x_202 = x_195 = None
        x_204 = torch.conv2d(
            x_203,
            l_self_modules_blocks_modules_4_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_203 = (
            l_self_modules_blocks_modules_4_modules_0_modules_conv_parameters_weight_
        ) = None
        x_205 = torch.nn.functional.batch_norm(
            x_204,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_204 = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_
        ) = (
            l_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_
        ) = None
        x_206 = torch.nn.functional.relu(x_205, inplace=True)
        x_205 = None
        x_207 = torch.nn.functional.adaptive_avg_pool2d(x_206, 1)
        x_206 = None
        x_208 = torch.conv2d(
            x_207,
            l_self_modules_conv_head_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_207 = l_self_modules_conv_head_parameters_weight_ = None
        x_209 = torch.nn.functional.batch_norm(
            x_208,
            l_self_modules_norm_head_buffers_running_mean_,
            l_self_modules_norm_head_buffers_running_var_,
            l_self_modules_norm_head_parameters_weight_,
            l_self_modules_norm_head_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_208 = (
            l_self_modules_norm_head_buffers_running_mean_
        ) = (
            l_self_modules_norm_head_buffers_running_var_
        ) = (
            l_self_modules_norm_head_parameters_weight_
        ) = l_self_modules_norm_head_parameters_bias_ = None
        x_210 = torch.nn.functional.relu(x_209, inplace=True)
        x_209 = None
        x_211 = x_210.flatten(1, -1)
        x_210 = None
        x_212 = torch._C._nn.linear(
            x_211,
            l_self_modules_classifier_parameters_weight_,
            l_self_modules_classifier_parameters_bias_,
        )
        x_211 = (
            l_self_modules_classifier_parameters_weight_
        ) = l_self_modules_classifier_parameters_bias_ = None
        return (x_212,)
