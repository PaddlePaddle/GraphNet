import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv_dw_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv_dw_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_0_modules_conv_dw_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_0_modules_conv_dw_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv_dw_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv_pwl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv_pwl_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_0_modules_conv_pwl_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_0_modules_conv_pwl_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_conv_pwl_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv_dw_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv_dw_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv_dw_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv_dw_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv_dw_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv_pwl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv_pwl_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv_pwl_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv_pwl_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv_pwl_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_dw_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_dw_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv_dw_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv_dw_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_dw_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_pwl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_pwl_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv_pwl_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv_pwl_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_pwl_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_dw_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_dw_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv_dw_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv_dw_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_dw_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_se_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_se_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_se_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_se_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_pwl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_pwl_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv_pwl_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv_pwl_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_pwl_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv_dw_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv_dw_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv_dw_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv_dw_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv_dw_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_se_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_se_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_se_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_se_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv_pwl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv_pwl_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv_pwl_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv_pwl_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv_pwl_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_dw_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_dw_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv_dw_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv_dw_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_dw_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_se_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_se_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_se_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_se_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_pwl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_pwl_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv_pwl_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv_pwl_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_pwl_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_conv_dw_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_conv_dw_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_conv_dw_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_conv_dw_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_conv_dw_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_se_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_se_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_se_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_se_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_conv_pwl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_conv_pwl_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_conv_pwl_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_conv_pwl_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_conv_pwl_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_conv_dw_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_conv_dw_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_conv_dw_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_conv_dw_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_conv_dw_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_se_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_se_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_se_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_se_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_conv_pwl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_conv_pwl_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_conv_pwl_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_conv_pwl_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_conv_pwl_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_conv_dw_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_conv_dw_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_modules_conv_dw_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_modules_conv_dw_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_conv_dw_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_se_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_modules_se_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_modules_se_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_se_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_conv_pwl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_conv_pwl_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_modules_conv_pwl_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_modules_conv_pwl_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_conv_pwl_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_9_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_9_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_conv_dw_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_conv_dw_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_9_modules_conv_dw_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_9_modules_conv_dw_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_conv_dw_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_se_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_9_modules_se_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_9_modules_se_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_se_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_conv_pwl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_conv_pwl_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_9_modules_conv_pwl_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_9_modules_conv_pwl_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_conv_pwl_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_10_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_10_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_conv_dw_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_conv_dw_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_10_modules_conv_dw_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_10_modules_conv_dw_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_conv_dw_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_se_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_10_modules_se_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_10_modules_se_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_se_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_conv_pwl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_conv_pwl_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_10_modules_conv_pwl_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_10_modules_conv_pwl_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_conv_pwl_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_conv_dw_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_conv_dw_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_conv_dw_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_conv_dw_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_conv_dw_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_se_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_se_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_se_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_se_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_conv_pwl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_conv_pwl_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_conv_pwl_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_conv_pwl_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_conv_pwl_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_conv_dw_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_conv_dw_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_modules_conv_dw_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_modules_conv_dw_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_conv_dw_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_se_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_modules_se_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_modules_se_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_se_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_conv_pwl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_conv_pwl_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_modules_conv_pwl_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_modules_conv_pwl_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_conv_pwl_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_13_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_13_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_conv_dw_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_conv_dw_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_13_modules_conv_dw_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_13_modules_conv_dw_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_conv_dw_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_se_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_13_modules_se_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_13_modules_se_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_se_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_conv_pwl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_conv_pwl_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_13_modules_conv_pwl_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_13_modules_conv_pwl_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_conv_pwl_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_14_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_14_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_conv_dw_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_conv_dw_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_14_modules_conv_dw_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_14_modules_conv_dw_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_conv_dw_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_se_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_14_modules_se_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_14_modules_se_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_se_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_conv_pwl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_conv_pwl_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_14_modules_conv_pwl_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_14_modules_conv_pwl_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_conv_pwl_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_15_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_15_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_conv_dw_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_conv_dw_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_15_modules_conv_dw_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_15_modules_conv_dw_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_conv_dw_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_se_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_15_modules_se_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_15_modules_se_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_se_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_conv_pwl_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_conv_pwl_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_15_modules_conv_pwl_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_15_modules_conv_pwl_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_conv_pwl_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_16_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_16_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_0_modules_conv_dw_modules_conv_parameters_weight_ = L_self_modules_features_modules_0_modules_conv_dw_modules_conv_parameters_weight_
        l_self_modules_features_modules_0_modules_conv_dw_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_0_modules_conv_dw_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_0_modules_conv_dw_modules_bn_buffers_running_var_ = L_self_modules_features_modules_0_modules_conv_dw_modules_bn_buffers_running_var_
        l_self_modules_features_modules_0_modules_conv_dw_modules_bn_parameters_weight_ = L_self_modules_features_modules_0_modules_conv_dw_modules_bn_parameters_weight_
        l_self_modules_features_modules_0_modules_conv_dw_modules_bn_parameters_bias_ = L_self_modules_features_modules_0_modules_conv_dw_modules_bn_parameters_bias_
        l_self_modules_features_modules_0_modules_conv_pwl_modules_conv_parameters_weight_ = L_self_modules_features_modules_0_modules_conv_pwl_modules_conv_parameters_weight_
        l_self_modules_features_modules_0_modules_conv_pwl_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_0_modules_conv_pwl_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_0_modules_conv_pwl_modules_bn_buffers_running_var_ = L_self_modules_features_modules_0_modules_conv_pwl_modules_bn_buffers_running_var_
        l_self_modules_features_modules_0_modules_conv_pwl_modules_bn_parameters_weight_ = L_self_modules_features_modules_0_modules_conv_pwl_modules_bn_parameters_weight_
        l_self_modules_features_modules_0_modules_conv_pwl_modules_bn_parameters_bias_ = L_self_modules_features_modules_0_modules_conv_pwl_modules_bn_parameters_bias_
        l_self_modules_features_modules_1_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_features_modules_1_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_features_modules_1_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_1_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_1_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_features_modules_1_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_features_modules_1_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_features_modules_1_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_features_modules_1_modules_conv_exp_modules_bn_parameters_bias_ = L_self_modules_features_modules_1_modules_conv_exp_modules_bn_parameters_bias_
        l_self_modules_features_modules_1_modules_conv_dw_modules_conv_parameters_weight_ = L_self_modules_features_modules_1_modules_conv_dw_modules_conv_parameters_weight_
        l_self_modules_features_modules_1_modules_conv_dw_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_1_modules_conv_dw_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_1_modules_conv_dw_modules_bn_buffers_running_var_ = L_self_modules_features_modules_1_modules_conv_dw_modules_bn_buffers_running_var_
        l_self_modules_features_modules_1_modules_conv_dw_modules_bn_parameters_weight_ = L_self_modules_features_modules_1_modules_conv_dw_modules_bn_parameters_weight_
        l_self_modules_features_modules_1_modules_conv_dw_modules_bn_parameters_bias_ = L_self_modules_features_modules_1_modules_conv_dw_modules_bn_parameters_bias_
        l_self_modules_features_modules_1_modules_conv_pwl_modules_conv_parameters_weight_ = L_self_modules_features_modules_1_modules_conv_pwl_modules_conv_parameters_weight_
        l_self_modules_features_modules_1_modules_conv_pwl_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_1_modules_conv_pwl_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_1_modules_conv_pwl_modules_bn_buffers_running_var_ = L_self_modules_features_modules_1_modules_conv_pwl_modules_bn_buffers_running_var_
        l_self_modules_features_modules_1_modules_conv_pwl_modules_bn_parameters_weight_ = L_self_modules_features_modules_1_modules_conv_pwl_modules_bn_parameters_weight_
        l_self_modules_features_modules_1_modules_conv_pwl_modules_bn_parameters_bias_ = L_self_modules_features_modules_1_modules_conv_pwl_modules_bn_parameters_bias_
        l_self_modules_features_modules_2_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_features_modules_2_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_features_modules_2_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_2_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_2_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_features_modules_2_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_features_modules_2_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_features_modules_2_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_features_modules_2_modules_conv_exp_modules_bn_parameters_bias_ = L_self_modules_features_modules_2_modules_conv_exp_modules_bn_parameters_bias_
        l_self_modules_features_modules_2_modules_conv_dw_modules_conv_parameters_weight_ = L_self_modules_features_modules_2_modules_conv_dw_modules_conv_parameters_weight_
        l_self_modules_features_modules_2_modules_conv_dw_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_2_modules_conv_dw_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_2_modules_conv_dw_modules_bn_buffers_running_var_ = L_self_modules_features_modules_2_modules_conv_dw_modules_bn_buffers_running_var_
        l_self_modules_features_modules_2_modules_conv_dw_modules_bn_parameters_weight_ = L_self_modules_features_modules_2_modules_conv_dw_modules_bn_parameters_weight_
        l_self_modules_features_modules_2_modules_conv_dw_modules_bn_parameters_bias_ = L_self_modules_features_modules_2_modules_conv_dw_modules_bn_parameters_bias_
        l_self_modules_features_modules_2_modules_conv_pwl_modules_conv_parameters_weight_ = L_self_modules_features_modules_2_modules_conv_pwl_modules_conv_parameters_weight_
        l_self_modules_features_modules_2_modules_conv_pwl_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_2_modules_conv_pwl_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_2_modules_conv_pwl_modules_bn_buffers_running_var_ = L_self_modules_features_modules_2_modules_conv_pwl_modules_bn_buffers_running_var_
        l_self_modules_features_modules_2_modules_conv_pwl_modules_bn_parameters_weight_ = L_self_modules_features_modules_2_modules_conv_pwl_modules_bn_parameters_weight_
        l_self_modules_features_modules_2_modules_conv_pwl_modules_bn_parameters_bias_ = L_self_modules_features_modules_2_modules_conv_pwl_modules_bn_parameters_bias_
        l_self_modules_features_modules_3_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_features_modules_3_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_features_modules_3_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_3_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_3_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_features_modules_3_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_features_modules_3_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_features_modules_3_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_features_modules_3_modules_conv_exp_modules_bn_parameters_bias_ = L_self_modules_features_modules_3_modules_conv_exp_modules_bn_parameters_bias_
        l_self_modules_features_modules_3_modules_conv_dw_modules_conv_parameters_weight_ = L_self_modules_features_modules_3_modules_conv_dw_modules_conv_parameters_weight_
        l_self_modules_features_modules_3_modules_conv_dw_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_3_modules_conv_dw_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_3_modules_conv_dw_modules_bn_buffers_running_var_ = L_self_modules_features_modules_3_modules_conv_dw_modules_bn_buffers_running_var_
        l_self_modules_features_modules_3_modules_conv_dw_modules_bn_parameters_weight_ = L_self_modules_features_modules_3_modules_conv_dw_modules_bn_parameters_weight_
        l_self_modules_features_modules_3_modules_conv_dw_modules_bn_parameters_bias_ = L_self_modules_features_modules_3_modules_conv_dw_modules_bn_parameters_bias_
        l_self_modules_features_modules_3_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_se_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_3_modules_se_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_3_modules_se_modules_bn_buffers_running_var_ = (
            L_self_modules_features_modules_3_modules_se_modules_bn_buffers_running_var_
        )
        l_self_modules_features_modules_3_modules_se_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_se_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_se_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_se_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_conv_pwl_modules_conv_parameters_weight_ = L_self_modules_features_modules_3_modules_conv_pwl_modules_conv_parameters_weight_
        l_self_modules_features_modules_3_modules_conv_pwl_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_3_modules_conv_pwl_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_3_modules_conv_pwl_modules_bn_buffers_running_var_ = L_self_modules_features_modules_3_modules_conv_pwl_modules_bn_buffers_running_var_
        l_self_modules_features_modules_3_modules_conv_pwl_modules_bn_parameters_weight_ = L_self_modules_features_modules_3_modules_conv_pwl_modules_bn_parameters_weight_
        l_self_modules_features_modules_3_modules_conv_pwl_modules_bn_parameters_bias_ = L_self_modules_features_modules_3_modules_conv_pwl_modules_bn_parameters_bias_
        l_self_modules_features_modules_4_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_features_modules_4_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_features_modules_4_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_4_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_4_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_features_modules_4_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_features_modules_4_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_features_modules_4_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_features_modules_4_modules_conv_exp_modules_bn_parameters_bias_ = L_self_modules_features_modules_4_modules_conv_exp_modules_bn_parameters_bias_
        l_self_modules_features_modules_4_modules_conv_dw_modules_conv_parameters_weight_ = L_self_modules_features_modules_4_modules_conv_dw_modules_conv_parameters_weight_
        l_self_modules_features_modules_4_modules_conv_dw_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_4_modules_conv_dw_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_4_modules_conv_dw_modules_bn_buffers_running_var_ = L_self_modules_features_modules_4_modules_conv_dw_modules_bn_buffers_running_var_
        l_self_modules_features_modules_4_modules_conv_dw_modules_bn_parameters_weight_ = L_self_modules_features_modules_4_modules_conv_dw_modules_bn_parameters_weight_
        l_self_modules_features_modules_4_modules_conv_dw_modules_bn_parameters_bias_ = L_self_modules_features_modules_4_modules_conv_dw_modules_bn_parameters_bias_
        l_self_modules_features_modules_4_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_features_modules_4_modules_se_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_4_modules_se_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_4_modules_se_modules_bn_buffers_running_var_ = (
            L_self_modules_features_modules_4_modules_se_modules_bn_buffers_running_var_
        )
        l_self_modules_features_modules_4_modules_se_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_se_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_se_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_se_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_4_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_features_modules_4_modules_conv_pwl_modules_conv_parameters_weight_ = L_self_modules_features_modules_4_modules_conv_pwl_modules_conv_parameters_weight_
        l_self_modules_features_modules_4_modules_conv_pwl_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_4_modules_conv_pwl_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_4_modules_conv_pwl_modules_bn_buffers_running_var_ = L_self_modules_features_modules_4_modules_conv_pwl_modules_bn_buffers_running_var_
        l_self_modules_features_modules_4_modules_conv_pwl_modules_bn_parameters_weight_ = L_self_modules_features_modules_4_modules_conv_pwl_modules_bn_parameters_weight_
        l_self_modules_features_modules_4_modules_conv_pwl_modules_bn_parameters_bias_ = L_self_modules_features_modules_4_modules_conv_pwl_modules_bn_parameters_bias_
        l_self_modules_features_modules_5_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_features_modules_5_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_features_modules_5_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_5_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_5_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_features_modules_5_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_features_modules_5_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_features_modules_5_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_features_modules_5_modules_conv_exp_modules_bn_parameters_bias_ = L_self_modules_features_modules_5_modules_conv_exp_modules_bn_parameters_bias_
        l_self_modules_features_modules_5_modules_conv_dw_modules_conv_parameters_weight_ = L_self_modules_features_modules_5_modules_conv_dw_modules_conv_parameters_weight_
        l_self_modules_features_modules_5_modules_conv_dw_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_5_modules_conv_dw_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_5_modules_conv_dw_modules_bn_buffers_running_var_ = L_self_modules_features_modules_5_modules_conv_dw_modules_bn_buffers_running_var_
        l_self_modules_features_modules_5_modules_conv_dw_modules_bn_parameters_weight_ = L_self_modules_features_modules_5_modules_conv_dw_modules_bn_parameters_weight_
        l_self_modules_features_modules_5_modules_conv_dw_modules_bn_parameters_bias_ = L_self_modules_features_modules_5_modules_conv_dw_modules_bn_parameters_bias_
        l_self_modules_features_modules_5_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_se_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_5_modules_se_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_5_modules_se_modules_bn_buffers_running_var_ = (
            L_self_modules_features_modules_5_modules_se_modules_bn_buffers_running_var_
        )
        l_self_modules_features_modules_5_modules_se_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_se_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_se_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_se_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_conv_pwl_modules_conv_parameters_weight_ = L_self_modules_features_modules_5_modules_conv_pwl_modules_conv_parameters_weight_
        l_self_modules_features_modules_5_modules_conv_pwl_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_5_modules_conv_pwl_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_5_modules_conv_pwl_modules_bn_buffers_running_var_ = L_self_modules_features_modules_5_modules_conv_pwl_modules_bn_buffers_running_var_
        l_self_modules_features_modules_5_modules_conv_pwl_modules_bn_parameters_weight_ = L_self_modules_features_modules_5_modules_conv_pwl_modules_bn_parameters_weight_
        l_self_modules_features_modules_5_modules_conv_pwl_modules_bn_parameters_bias_ = L_self_modules_features_modules_5_modules_conv_pwl_modules_bn_parameters_bias_
        l_self_modules_features_modules_6_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_features_modules_6_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_features_modules_6_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_6_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_6_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_features_modules_6_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_features_modules_6_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_features_modules_6_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_features_modules_6_modules_conv_exp_modules_bn_parameters_bias_ = L_self_modules_features_modules_6_modules_conv_exp_modules_bn_parameters_bias_
        l_self_modules_features_modules_6_modules_conv_dw_modules_conv_parameters_weight_ = L_self_modules_features_modules_6_modules_conv_dw_modules_conv_parameters_weight_
        l_self_modules_features_modules_6_modules_conv_dw_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_6_modules_conv_dw_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_6_modules_conv_dw_modules_bn_buffers_running_var_ = L_self_modules_features_modules_6_modules_conv_dw_modules_bn_buffers_running_var_
        l_self_modules_features_modules_6_modules_conv_dw_modules_bn_parameters_weight_ = L_self_modules_features_modules_6_modules_conv_dw_modules_bn_parameters_weight_
        l_self_modules_features_modules_6_modules_conv_dw_modules_bn_parameters_bias_ = L_self_modules_features_modules_6_modules_conv_dw_modules_bn_parameters_bias_
        l_self_modules_features_modules_6_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_features_modules_6_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_features_modules_6_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_features_modules_6_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_features_modules_6_modules_se_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_6_modules_se_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_6_modules_se_modules_bn_buffers_running_var_ = (
            L_self_modules_features_modules_6_modules_se_modules_bn_buffers_running_var_
        )
        l_self_modules_features_modules_6_modules_se_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_6_modules_se_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_6_modules_se_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_6_modules_se_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_6_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_features_modules_6_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_features_modules_6_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_features_modules_6_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_features_modules_6_modules_conv_pwl_modules_conv_parameters_weight_ = L_self_modules_features_modules_6_modules_conv_pwl_modules_conv_parameters_weight_
        l_self_modules_features_modules_6_modules_conv_pwl_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_6_modules_conv_pwl_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_6_modules_conv_pwl_modules_bn_buffers_running_var_ = L_self_modules_features_modules_6_modules_conv_pwl_modules_bn_buffers_running_var_
        l_self_modules_features_modules_6_modules_conv_pwl_modules_bn_parameters_weight_ = L_self_modules_features_modules_6_modules_conv_pwl_modules_bn_parameters_weight_
        l_self_modules_features_modules_6_modules_conv_pwl_modules_bn_parameters_bias_ = L_self_modules_features_modules_6_modules_conv_pwl_modules_bn_parameters_bias_
        l_self_modules_features_modules_7_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_features_modules_7_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_features_modules_7_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_7_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_7_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_features_modules_7_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_features_modules_7_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_features_modules_7_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_features_modules_7_modules_conv_exp_modules_bn_parameters_bias_ = L_self_modules_features_modules_7_modules_conv_exp_modules_bn_parameters_bias_
        l_self_modules_features_modules_7_modules_conv_dw_modules_conv_parameters_weight_ = L_self_modules_features_modules_7_modules_conv_dw_modules_conv_parameters_weight_
        l_self_modules_features_modules_7_modules_conv_dw_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_7_modules_conv_dw_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_7_modules_conv_dw_modules_bn_buffers_running_var_ = L_self_modules_features_modules_7_modules_conv_dw_modules_bn_buffers_running_var_
        l_self_modules_features_modules_7_modules_conv_dw_modules_bn_parameters_weight_ = L_self_modules_features_modules_7_modules_conv_dw_modules_bn_parameters_weight_
        l_self_modules_features_modules_7_modules_conv_dw_modules_bn_parameters_bias_ = L_self_modules_features_modules_7_modules_conv_dw_modules_bn_parameters_bias_
        l_self_modules_features_modules_7_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_features_modules_7_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_features_modules_7_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_features_modules_7_modules_se_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_7_modules_se_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_7_modules_se_modules_bn_buffers_running_var_ = (
            L_self_modules_features_modules_7_modules_se_modules_bn_buffers_running_var_
        )
        l_self_modules_features_modules_7_modules_se_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_7_modules_se_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_se_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_7_modules_se_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_7_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_features_modules_7_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_features_modules_7_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_features_modules_7_modules_conv_pwl_modules_conv_parameters_weight_ = L_self_modules_features_modules_7_modules_conv_pwl_modules_conv_parameters_weight_
        l_self_modules_features_modules_7_modules_conv_pwl_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_7_modules_conv_pwl_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_7_modules_conv_pwl_modules_bn_buffers_running_var_ = L_self_modules_features_modules_7_modules_conv_pwl_modules_bn_buffers_running_var_
        l_self_modules_features_modules_7_modules_conv_pwl_modules_bn_parameters_weight_ = L_self_modules_features_modules_7_modules_conv_pwl_modules_bn_parameters_weight_
        l_self_modules_features_modules_7_modules_conv_pwl_modules_bn_parameters_bias_ = L_self_modules_features_modules_7_modules_conv_pwl_modules_bn_parameters_bias_
        l_self_modules_features_modules_8_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_features_modules_8_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_features_modules_8_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_8_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_8_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_features_modules_8_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_features_modules_8_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_features_modules_8_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_features_modules_8_modules_conv_exp_modules_bn_parameters_bias_ = L_self_modules_features_modules_8_modules_conv_exp_modules_bn_parameters_bias_
        l_self_modules_features_modules_8_modules_conv_dw_modules_conv_parameters_weight_ = L_self_modules_features_modules_8_modules_conv_dw_modules_conv_parameters_weight_
        l_self_modules_features_modules_8_modules_conv_dw_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_8_modules_conv_dw_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_8_modules_conv_dw_modules_bn_buffers_running_var_ = L_self_modules_features_modules_8_modules_conv_dw_modules_bn_buffers_running_var_
        l_self_modules_features_modules_8_modules_conv_dw_modules_bn_parameters_weight_ = L_self_modules_features_modules_8_modules_conv_dw_modules_bn_parameters_weight_
        l_self_modules_features_modules_8_modules_conv_dw_modules_bn_parameters_bias_ = L_self_modules_features_modules_8_modules_conv_dw_modules_bn_parameters_bias_
        l_self_modules_features_modules_8_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_features_modules_8_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_features_modules_8_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_features_modules_8_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_features_modules_8_modules_se_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_8_modules_se_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_8_modules_se_modules_bn_buffers_running_var_ = (
            L_self_modules_features_modules_8_modules_se_modules_bn_buffers_running_var_
        )
        l_self_modules_features_modules_8_modules_se_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_8_modules_se_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_8_modules_se_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_8_modules_se_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_8_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_features_modules_8_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_features_modules_8_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_features_modules_8_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_features_modules_8_modules_conv_pwl_modules_conv_parameters_weight_ = L_self_modules_features_modules_8_modules_conv_pwl_modules_conv_parameters_weight_
        l_self_modules_features_modules_8_modules_conv_pwl_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_8_modules_conv_pwl_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_8_modules_conv_pwl_modules_bn_buffers_running_var_ = L_self_modules_features_modules_8_modules_conv_pwl_modules_bn_buffers_running_var_
        l_self_modules_features_modules_8_modules_conv_pwl_modules_bn_parameters_weight_ = L_self_modules_features_modules_8_modules_conv_pwl_modules_bn_parameters_weight_
        l_self_modules_features_modules_8_modules_conv_pwl_modules_bn_parameters_bias_ = L_self_modules_features_modules_8_modules_conv_pwl_modules_bn_parameters_bias_
        l_self_modules_features_modules_9_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_features_modules_9_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_features_modules_9_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_9_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_9_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_features_modules_9_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_features_modules_9_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_features_modules_9_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_features_modules_9_modules_conv_exp_modules_bn_parameters_bias_ = L_self_modules_features_modules_9_modules_conv_exp_modules_bn_parameters_bias_
        l_self_modules_features_modules_9_modules_conv_dw_modules_conv_parameters_weight_ = L_self_modules_features_modules_9_modules_conv_dw_modules_conv_parameters_weight_
        l_self_modules_features_modules_9_modules_conv_dw_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_9_modules_conv_dw_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_9_modules_conv_dw_modules_bn_buffers_running_var_ = L_self_modules_features_modules_9_modules_conv_dw_modules_bn_buffers_running_var_
        l_self_modules_features_modules_9_modules_conv_dw_modules_bn_parameters_weight_ = L_self_modules_features_modules_9_modules_conv_dw_modules_bn_parameters_weight_
        l_self_modules_features_modules_9_modules_conv_dw_modules_bn_parameters_bias_ = L_self_modules_features_modules_9_modules_conv_dw_modules_bn_parameters_bias_
        l_self_modules_features_modules_9_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_features_modules_9_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_features_modules_9_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_features_modules_9_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_features_modules_9_modules_se_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_9_modules_se_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_9_modules_se_modules_bn_buffers_running_var_ = (
            L_self_modules_features_modules_9_modules_se_modules_bn_buffers_running_var_
        )
        l_self_modules_features_modules_9_modules_se_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_9_modules_se_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_9_modules_se_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_9_modules_se_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_9_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_features_modules_9_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_features_modules_9_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_features_modules_9_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_features_modules_9_modules_conv_pwl_modules_conv_parameters_weight_ = L_self_modules_features_modules_9_modules_conv_pwl_modules_conv_parameters_weight_
        l_self_modules_features_modules_9_modules_conv_pwl_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_9_modules_conv_pwl_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_9_modules_conv_pwl_modules_bn_buffers_running_var_ = L_self_modules_features_modules_9_modules_conv_pwl_modules_bn_buffers_running_var_
        l_self_modules_features_modules_9_modules_conv_pwl_modules_bn_parameters_weight_ = L_self_modules_features_modules_9_modules_conv_pwl_modules_bn_parameters_weight_
        l_self_modules_features_modules_9_modules_conv_pwl_modules_bn_parameters_bias_ = L_self_modules_features_modules_9_modules_conv_pwl_modules_bn_parameters_bias_
        l_self_modules_features_modules_10_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_features_modules_10_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_features_modules_10_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_10_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_10_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_features_modules_10_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_features_modules_10_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_features_modules_10_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_features_modules_10_modules_conv_exp_modules_bn_parameters_bias_ = L_self_modules_features_modules_10_modules_conv_exp_modules_bn_parameters_bias_
        l_self_modules_features_modules_10_modules_conv_dw_modules_conv_parameters_weight_ = L_self_modules_features_modules_10_modules_conv_dw_modules_conv_parameters_weight_
        l_self_modules_features_modules_10_modules_conv_dw_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_10_modules_conv_dw_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_10_modules_conv_dw_modules_bn_buffers_running_var_ = L_self_modules_features_modules_10_modules_conv_dw_modules_bn_buffers_running_var_
        l_self_modules_features_modules_10_modules_conv_dw_modules_bn_parameters_weight_ = L_self_modules_features_modules_10_modules_conv_dw_modules_bn_parameters_weight_
        l_self_modules_features_modules_10_modules_conv_dw_modules_bn_parameters_bias_ = L_self_modules_features_modules_10_modules_conv_dw_modules_bn_parameters_bias_
        l_self_modules_features_modules_10_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_features_modules_10_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_features_modules_10_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_features_modules_10_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_features_modules_10_modules_se_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_10_modules_se_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_10_modules_se_modules_bn_buffers_running_var_ = L_self_modules_features_modules_10_modules_se_modules_bn_buffers_running_var_
        l_self_modules_features_modules_10_modules_se_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_10_modules_se_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_10_modules_se_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_10_modules_se_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_10_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_features_modules_10_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_features_modules_10_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_features_modules_10_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_features_modules_10_modules_conv_pwl_modules_conv_parameters_weight_ = L_self_modules_features_modules_10_modules_conv_pwl_modules_conv_parameters_weight_
        l_self_modules_features_modules_10_modules_conv_pwl_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_10_modules_conv_pwl_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_10_modules_conv_pwl_modules_bn_buffers_running_var_ = L_self_modules_features_modules_10_modules_conv_pwl_modules_bn_buffers_running_var_
        l_self_modules_features_modules_10_modules_conv_pwl_modules_bn_parameters_weight_ = L_self_modules_features_modules_10_modules_conv_pwl_modules_bn_parameters_weight_
        l_self_modules_features_modules_10_modules_conv_pwl_modules_bn_parameters_bias_ = L_self_modules_features_modules_10_modules_conv_pwl_modules_bn_parameters_bias_
        l_self_modules_features_modules_11_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_features_modules_11_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_features_modules_11_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_11_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_11_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_features_modules_11_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_features_modules_11_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_features_modules_11_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_features_modules_11_modules_conv_exp_modules_bn_parameters_bias_ = L_self_modules_features_modules_11_modules_conv_exp_modules_bn_parameters_bias_
        l_self_modules_features_modules_11_modules_conv_dw_modules_conv_parameters_weight_ = L_self_modules_features_modules_11_modules_conv_dw_modules_conv_parameters_weight_
        l_self_modules_features_modules_11_modules_conv_dw_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_11_modules_conv_dw_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_11_modules_conv_dw_modules_bn_buffers_running_var_ = L_self_modules_features_modules_11_modules_conv_dw_modules_bn_buffers_running_var_
        l_self_modules_features_modules_11_modules_conv_dw_modules_bn_parameters_weight_ = L_self_modules_features_modules_11_modules_conv_dw_modules_bn_parameters_weight_
        l_self_modules_features_modules_11_modules_conv_dw_modules_bn_parameters_bias_ = L_self_modules_features_modules_11_modules_conv_dw_modules_bn_parameters_bias_
        l_self_modules_features_modules_11_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_features_modules_11_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_features_modules_11_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_features_modules_11_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_features_modules_11_modules_se_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_11_modules_se_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_11_modules_se_modules_bn_buffers_running_var_ = L_self_modules_features_modules_11_modules_se_modules_bn_buffers_running_var_
        l_self_modules_features_modules_11_modules_se_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_11_modules_se_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_11_modules_se_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_11_modules_se_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_11_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_features_modules_11_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_features_modules_11_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_features_modules_11_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_features_modules_11_modules_conv_pwl_modules_conv_parameters_weight_ = L_self_modules_features_modules_11_modules_conv_pwl_modules_conv_parameters_weight_
        l_self_modules_features_modules_11_modules_conv_pwl_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_11_modules_conv_pwl_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_11_modules_conv_pwl_modules_bn_buffers_running_var_ = L_self_modules_features_modules_11_modules_conv_pwl_modules_bn_buffers_running_var_
        l_self_modules_features_modules_11_modules_conv_pwl_modules_bn_parameters_weight_ = L_self_modules_features_modules_11_modules_conv_pwl_modules_bn_parameters_weight_
        l_self_modules_features_modules_11_modules_conv_pwl_modules_bn_parameters_bias_ = L_self_modules_features_modules_11_modules_conv_pwl_modules_bn_parameters_bias_
        l_self_modules_features_modules_12_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_features_modules_12_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_features_modules_12_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_12_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_12_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_features_modules_12_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_features_modules_12_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_features_modules_12_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_features_modules_12_modules_conv_exp_modules_bn_parameters_bias_ = L_self_modules_features_modules_12_modules_conv_exp_modules_bn_parameters_bias_
        l_self_modules_features_modules_12_modules_conv_dw_modules_conv_parameters_weight_ = L_self_modules_features_modules_12_modules_conv_dw_modules_conv_parameters_weight_
        l_self_modules_features_modules_12_modules_conv_dw_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_12_modules_conv_dw_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_12_modules_conv_dw_modules_bn_buffers_running_var_ = L_self_modules_features_modules_12_modules_conv_dw_modules_bn_buffers_running_var_
        l_self_modules_features_modules_12_modules_conv_dw_modules_bn_parameters_weight_ = L_self_modules_features_modules_12_modules_conv_dw_modules_bn_parameters_weight_
        l_self_modules_features_modules_12_modules_conv_dw_modules_bn_parameters_bias_ = L_self_modules_features_modules_12_modules_conv_dw_modules_bn_parameters_bias_
        l_self_modules_features_modules_12_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_features_modules_12_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_features_modules_12_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_features_modules_12_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_features_modules_12_modules_se_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_12_modules_se_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_12_modules_se_modules_bn_buffers_running_var_ = L_self_modules_features_modules_12_modules_se_modules_bn_buffers_running_var_
        l_self_modules_features_modules_12_modules_se_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_12_modules_se_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_12_modules_se_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_12_modules_se_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_12_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_features_modules_12_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_features_modules_12_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_features_modules_12_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_features_modules_12_modules_conv_pwl_modules_conv_parameters_weight_ = L_self_modules_features_modules_12_modules_conv_pwl_modules_conv_parameters_weight_
        l_self_modules_features_modules_12_modules_conv_pwl_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_12_modules_conv_pwl_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_12_modules_conv_pwl_modules_bn_buffers_running_var_ = L_self_modules_features_modules_12_modules_conv_pwl_modules_bn_buffers_running_var_
        l_self_modules_features_modules_12_modules_conv_pwl_modules_bn_parameters_weight_ = L_self_modules_features_modules_12_modules_conv_pwl_modules_bn_parameters_weight_
        l_self_modules_features_modules_12_modules_conv_pwl_modules_bn_parameters_bias_ = L_self_modules_features_modules_12_modules_conv_pwl_modules_bn_parameters_bias_
        l_self_modules_features_modules_13_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_features_modules_13_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_features_modules_13_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_13_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_13_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_features_modules_13_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_features_modules_13_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_features_modules_13_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_features_modules_13_modules_conv_exp_modules_bn_parameters_bias_ = L_self_modules_features_modules_13_modules_conv_exp_modules_bn_parameters_bias_
        l_self_modules_features_modules_13_modules_conv_dw_modules_conv_parameters_weight_ = L_self_modules_features_modules_13_modules_conv_dw_modules_conv_parameters_weight_
        l_self_modules_features_modules_13_modules_conv_dw_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_13_modules_conv_dw_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_13_modules_conv_dw_modules_bn_buffers_running_var_ = L_self_modules_features_modules_13_modules_conv_dw_modules_bn_buffers_running_var_
        l_self_modules_features_modules_13_modules_conv_dw_modules_bn_parameters_weight_ = L_self_modules_features_modules_13_modules_conv_dw_modules_bn_parameters_weight_
        l_self_modules_features_modules_13_modules_conv_dw_modules_bn_parameters_bias_ = L_self_modules_features_modules_13_modules_conv_dw_modules_bn_parameters_bias_
        l_self_modules_features_modules_13_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_features_modules_13_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_features_modules_13_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_features_modules_13_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_features_modules_13_modules_se_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_13_modules_se_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_13_modules_se_modules_bn_buffers_running_var_ = L_self_modules_features_modules_13_modules_se_modules_bn_buffers_running_var_
        l_self_modules_features_modules_13_modules_se_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_13_modules_se_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_13_modules_se_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_13_modules_se_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_13_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_features_modules_13_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_features_modules_13_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_features_modules_13_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_features_modules_13_modules_conv_pwl_modules_conv_parameters_weight_ = L_self_modules_features_modules_13_modules_conv_pwl_modules_conv_parameters_weight_
        l_self_modules_features_modules_13_modules_conv_pwl_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_13_modules_conv_pwl_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_13_modules_conv_pwl_modules_bn_buffers_running_var_ = L_self_modules_features_modules_13_modules_conv_pwl_modules_bn_buffers_running_var_
        l_self_modules_features_modules_13_modules_conv_pwl_modules_bn_parameters_weight_ = L_self_modules_features_modules_13_modules_conv_pwl_modules_bn_parameters_weight_
        l_self_modules_features_modules_13_modules_conv_pwl_modules_bn_parameters_bias_ = L_self_modules_features_modules_13_modules_conv_pwl_modules_bn_parameters_bias_
        l_self_modules_features_modules_14_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_features_modules_14_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_features_modules_14_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_14_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_14_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_features_modules_14_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_features_modules_14_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_features_modules_14_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_features_modules_14_modules_conv_exp_modules_bn_parameters_bias_ = L_self_modules_features_modules_14_modules_conv_exp_modules_bn_parameters_bias_
        l_self_modules_features_modules_14_modules_conv_dw_modules_conv_parameters_weight_ = L_self_modules_features_modules_14_modules_conv_dw_modules_conv_parameters_weight_
        l_self_modules_features_modules_14_modules_conv_dw_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_14_modules_conv_dw_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_14_modules_conv_dw_modules_bn_buffers_running_var_ = L_self_modules_features_modules_14_modules_conv_dw_modules_bn_buffers_running_var_
        l_self_modules_features_modules_14_modules_conv_dw_modules_bn_parameters_weight_ = L_self_modules_features_modules_14_modules_conv_dw_modules_bn_parameters_weight_
        l_self_modules_features_modules_14_modules_conv_dw_modules_bn_parameters_bias_ = L_self_modules_features_modules_14_modules_conv_dw_modules_bn_parameters_bias_
        l_self_modules_features_modules_14_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_features_modules_14_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_features_modules_14_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_features_modules_14_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_features_modules_14_modules_se_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_14_modules_se_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_14_modules_se_modules_bn_buffers_running_var_ = L_self_modules_features_modules_14_modules_se_modules_bn_buffers_running_var_
        l_self_modules_features_modules_14_modules_se_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_14_modules_se_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_14_modules_se_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_14_modules_se_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_14_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_features_modules_14_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_features_modules_14_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_features_modules_14_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_features_modules_14_modules_conv_pwl_modules_conv_parameters_weight_ = L_self_modules_features_modules_14_modules_conv_pwl_modules_conv_parameters_weight_
        l_self_modules_features_modules_14_modules_conv_pwl_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_14_modules_conv_pwl_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_14_modules_conv_pwl_modules_bn_buffers_running_var_ = L_self_modules_features_modules_14_modules_conv_pwl_modules_bn_buffers_running_var_
        l_self_modules_features_modules_14_modules_conv_pwl_modules_bn_parameters_weight_ = L_self_modules_features_modules_14_modules_conv_pwl_modules_bn_parameters_weight_
        l_self_modules_features_modules_14_modules_conv_pwl_modules_bn_parameters_bias_ = L_self_modules_features_modules_14_modules_conv_pwl_modules_bn_parameters_bias_
        l_self_modules_features_modules_15_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_features_modules_15_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_features_modules_15_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_15_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_15_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_features_modules_15_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_features_modules_15_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_features_modules_15_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_features_modules_15_modules_conv_exp_modules_bn_parameters_bias_ = L_self_modules_features_modules_15_modules_conv_exp_modules_bn_parameters_bias_
        l_self_modules_features_modules_15_modules_conv_dw_modules_conv_parameters_weight_ = L_self_modules_features_modules_15_modules_conv_dw_modules_conv_parameters_weight_
        l_self_modules_features_modules_15_modules_conv_dw_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_15_modules_conv_dw_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_15_modules_conv_dw_modules_bn_buffers_running_var_ = L_self_modules_features_modules_15_modules_conv_dw_modules_bn_buffers_running_var_
        l_self_modules_features_modules_15_modules_conv_dw_modules_bn_parameters_weight_ = L_self_modules_features_modules_15_modules_conv_dw_modules_bn_parameters_weight_
        l_self_modules_features_modules_15_modules_conv_dw_modules_bn_parameters_bias_ = L_self_modules_features_modules_15_modules_conv_dw_modules_bn_parameters_bias_
        l_self_modules_features_modules_15_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_features_modules_15_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_features_modules_15_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_features_modules_15_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_features_modules_15_modules_se_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_15_modules_se_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_15_modules_se_modules_bn_buffers_running_var_ = L_self_modules_features_modules_15_modules_se_modules_bn_buffers_running_var_
        l_self_modules_features_modules_15_modules_se_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_15_modules_se_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_15_modules_se_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_15_modules_se_modules_bn_parameters_bias_
        )
        l_self_modules_features_modules_15_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_features_modules_15_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_features_modules_15_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_features_modules_15_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_features_modules_15_modules_conv_pwl_modules_conv_parameters_weight_ = L_self_modules_features_modules_15_modules_conv_pwl_modules_conv_parameters_weight_
        l_self_modules_features_modules_15_modules_conv_pwl_modules_bn_buffers_running_mean_ = L_self_modules_features_modules_15_modules_conv_pwl_modules_bn_buffers_running_mean_
        l_self_modules_features_modules_15_modules_conv_pwl_modules_bn_buffers_running_var_ = L_self_modules_features_modules_15_modules_conv_pwl_modules_bn_buffers_running_var_
        l_self_modules_features_modules_15_modules_conv_pwl_modules_bn_parameters_weight_ = L_self_modules_features_modules_15_modules_conv_pwl_modules_bn_parameters_weight_
        l_self_modules_features_modules_15_modules_conv_pwl_modules_bn_parameters_bias_ = L_self_modules_features_modules_15_modules_conv_pwl_modules_bn_parameters_bias_
        l_self_modules_features_modules_16_modules_conv_parameters_weight_ = (
            L_self_modules_features_modules_16_modules_conv_parameters_weight_
        )
        l_self_modules_features_modules_16_modules_bn_buffers_running_mean_ = (
            L_self_modules_features_modules_16_modules_bn_buffers_running_mean_
        )
        l_self_modules_features_modules_16_modules_bn_buffers_running_var_ = (
            L_self_modules_features_modules_16_modules_bn_buffers_running_var_
        )
        l_self_modules_features_modules_16_modules_bn_parameters_weight_ = (
            L_self_modules_features_modules_16_modules_bn_parameters_weight_
        )
        l_self_modules_features_modules_16_modules_bn_parameters_bias_ = (
            L_self_modules_features_modules_16_modules_bn_parameters_bias_
        )
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_modules_conv_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_stem_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_stem_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_bn_parameters_bias_ = None
        x_2 = torch.nn.functional.silu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_features_modules_0_modules_conv_dw_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            48,
        )
        x_2 = l_self_modules_features_modules_0_modules_conv_dw_modules_conv_parameters_weight_ = (None)
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_features_modules_0_modules_conv_dw_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_0_modules_conv_dw_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_0_modules_conv_dw_modules_bn_parameters_weight_,
            l_self_modules_features_modules_0_modules_conv_dw_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = l_self_modules_features_modules_0_modules_conv_dw_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_0_modules_conv_dw_modules_bn_buffers_running_var_ = l_self_modules_features_modules_0_modules_conv_dw_modules_bn_parameters_weight_ = l_self_modules_features_modules_0_modules_conv_dw_modules_bn_parameters_bias_ = (None)
        x_5 = torch.nn.functional.hardtanh(x_4, 0.0, 6.0, False)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_features_modules_0_modules_conv_pwl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_features_modules_0_modules_conv_pwl_modules_conv_parameters_weight_ = (None)
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_features_modules_0_modules_conv_pwl_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_0_modules_conv_pwl_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_0_modules_conv_pwl_modules_bn_parameters_weight_,
            l_self_modules_features_modules_0_modules_conv_pwl_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = l_self_modules_features_modules_0_modules_conv_pwl_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_0_modules_conv_pwl_modules_bn_buffers_running_var_ = l_self_modules_features_modules_0_modules_conv_pwl_modules_bn_parameters_weight_ = l_self_modules_features_modules_0_modules_conv_pwl_modules_bn_parameters_bias_ = (None)
        x_8 = torch.conv2d(
            x_7,
            l_self_modules_features_modules_1_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_7 = l_self_modules_features_modules_1_modules_conv_exp_modules_conv_parameters_weight_ = (None)
        x_9 = torch.nn.functional.batch_norm(
            x_8,
            l_self_modules_features_modules_1_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_1_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_features_modules_1_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_8 = l_self_modules_features_modules_1_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_1_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_features_modules_1_modules_conv_exp_modules_bn_parameters_weight_ = l_self_modules_features_modules_1_modules_conv_exp_modules_bn_parameters_bias_ = (None)
        x_10 = torch.nn.functional.silu(x_9, inplace=True)
        x_9 = None
        x_11 = torch.conv2d(
            x_10,
            l_self_modules_features_modules_1_modules_conv_dw_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            144,
        )
        x_10 = l_self_modules_features_modules_1_modules_conv_dw_modules_conv_parameters_weight_ = (None)
        x_12 = torch.nn.functional.batch_norm(
            x_11,
            l_self_modules_features_modules_1_modules_conv_dw_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_conv_dw_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_1_modules_conv_dw_modules_bn_parameters_weight_,
            l_self_modules_features_modules_1_modules_conv_dw_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_11 = l_self_modules_features_modules_1_modules_conv_dw_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_1_modules_conv_dw_modules_bn_buffers_running_var_ = l_self_modules_features_modules_1_modules_conv_dw_modules_bn_parameters_weight_ = l_self_modules_features_modules_1_modules_conv_dw_modules_bn_parameters_bias_ = (None)
        x_13 = torch.nn.functional.hardtanh(x_12, 0.0, 6.0, False)
        x_12 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_features_modules_1_modules_conv_pwl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_features_modules_1_modules_conv_pwl_modules_conv_parameters_weight_ = (None)
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_features_modules_1_modules_conv_pwl_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_conv_pwl_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_1_modules_conv_pwl_modules_bn_parameters_weight_,
            l_self_modules_features_modules_1_modules_conv_pwl_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_14 = l_self_modules_features_modules_1_modules_conv_pwl_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_1_modules_conv_pwl_modules_bn_buffers_running_var_ = l_self_modules_features_modules_1_modules_conv_pwl_modules_bn_parameters_weight_ = l_self_modules_features_modules_1_modules_conv_pwl_modules_bn_parameters_bias_ = (None)
        x_16 = torch.conv2d(
            x_15,
            l_self_modules_features_modules_2_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_2_modules_conv_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_features_modules_2_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_2_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_features_modules_2_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_features_modules_2_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_2_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_features_modules_2_modules_conv_exp_modules_bn_parameters_weight_ = l_self_modules_features_modules_2_modules_conv_exp_modules_bn_parameters_bias_ = (None)
        x_18 = torch.nn.functional.silu(x_17, inplace=True)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_features_modules_2_modules_conv_dw_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            246,
        )
        x_18 = l_self_modules_features_modules_2_modules_conv_dw_modules_conv_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_features_modules_2_modules_conv_dw_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_conv_dw_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_2_modules_conv_dw_modules_bn_parameters_weight_,
            l_self_modules_features_modules_2_modules_conv_dw_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_features_modules_2_modules_conv_dw_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_2_modules_conv_dw_modules_bn_buffers_running_var_ = l_self_modules_features_modules_2_modules_conv_dw_modules_bn_parameters_weight_ = l_self_modules_features_modules_2_modules_conv_dw_modules_bn_parameters_bias_ = (None)
        x_21 = torch.nn.functional.hardtanh(x_20, 0.0, 6.0, False)
        x_20 = None
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_features_modules_2_modules_conv_pwl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_21 = l_self_modules_features_modules_2_modules_conv_pwl_modules_conv_parameters_weight_ = (None)
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_features_modules_2_modules_conv_pwl_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_conv_pwl_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_2_modules_conv_pwl_modules_bn_parameters_weight_,
            l_self_modules_features_modules_2_modules_conv_pwl_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = l_self_modules_features_modules_2_modules_conv_pwl_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_2_modules_conv_pwl_modules_bn_buffers_running_var_ = l_self_modules_features_modules_2_modules_conv_pwl_modules_bn_parameters_weight_ = l_self_modules_features_modules_2_modules_conv_pwl_modules_bn_parameters_bias_ = (None)
        getitem = x_23[(slice(None, None, None), slice(0, 41, None))]
        add = getitem + x_15
        getitem = x_15 = None
        getitem_1 = x_23[(slice(None, None, None), slice(41, None, None))]
        x_23 = None
        x_24 = torch.cat([add, getitem_1], dim=1)
        add = getitem_1 = None
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_features_modules_3_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_24 = l_self_modules_features_modules_3_modules_conv_exp_modules_conv_parameters_weight_ = (None)
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_features_modules_3_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_3_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_features_modules_3_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = l_self_modules_features_modules_3_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_3_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_features_modules_3_modules_conv_exp_modules_bn_parameters_weight_ = l_self_modules_features_modules_3_modules_conv_exp_modules_bn_parameters_bias_ = (None)
        x_27 = torch.nn.functional.silu(x_26, inplace=True)
        x_26 = None
        x_28 = torch.conv2d(
            x_27,
            l_self_modules_features_modules_3_modules_conv_dw_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            348,
        )
        x_27 = l_self_modules_features_modules_3_modules_conv_dw_modules_conv_parameters_weight_ = (None)
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_features_modules_3_modules_conv_dw_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_conv_dw_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_3_modules_conv_dw_modules_bn_parameters_weight_,
            l_self_modules_features_modules_3_modules_conv_dw_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = l_self_modules_features_modules_3_modules_conv_dw_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_3_modules_conv_dw_modules_bn_buffers_running_var_ = l_self_modules_features_modules_3_modules_conv_dw_modules_bn_parameters_weight_ = l_self_modules_features_modules_3_modules_conv_dw_modules_bn_parameters_bias_ = (None)
        x_se = x_29.mean((2, 3), keepdim=True)
        x_se_1 = torch.conv2d(
            x_se,
            l_self_modules_features_modules_3_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_3_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se = (
            l_self_modules_features_modules_3_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_features_modules_3_modules_se_modules_fc1_parameters_bias_
        ) = None
        batch_norm_11 = torch.nn.functional.batch_norm(
            x_se_1,
            l_self_modules_features_modules_3_modules_se_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_se_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_3_modules_se_modules_bn_parameters_weight_,
            l_self_modules_features_modules_3_modules_se_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_se_1 = l_self_modules_features_modules_3_modules_se_modules_bn_buffers_running_mean_ = (
            l_self_modules_features_modules_3_modules_se_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_features_modules_3_modules_se_modules_bn_parameters_weight_
        ) = (
            l_self_modules_features_modules_3_modules_se_modules_bn_parameters_bias_
        ) = None
        x_se_2 = torch.nn.functional.relu(batch_norm_11, inplace=True)
        batch_norm_11 = None
        x_se_3 = torch.conv2d(
            x_se_2,
            l_self_modules_features_modules_3_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_3_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_2 = (
            l_self_modules_features_modules_3_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_features_modules_3_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid = x_se_3.sigmoid()
        x_se_3 = None
        x_30 = x_29 * sigmoid
        x_29 = sigmoid = None
        x_31 = torch.nn.functional.hardtanh(x_30, 0.0, 6.0, False)
        x_30 = None
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_features_modules_3_modules_conv_pwl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_31 = l_self_modules_features_modules_3_modules_conv_pwl_modules_conv_parameters_weight_ = (None)
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_features_modules_3_modules_conv_pwl_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_conv_pwl_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_3_modules_conv_pwl_modules_bn_parameters_weight_,
            l_self_modules_features_modules_3_modules_conv_pwl_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = l_self_modules_features_modules_3_modules_conv_pwl_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_3_modules_conv_pwl_modules_bn_buffers_running_var_ = l_self_modules_features_modules_3_modules_conv_pwl_modules_bn_parameters_weight_ = l_self_modules_features_modules_3_modules_conv_pwl_modules_bn_parameters_bias_ = (None)
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_features_modules_4_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_4_modules_conv_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_features_modules_4_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_4_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_features_modules_4_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = l_self_modules_features_modules_4_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_4_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_features_modules_4_modules_conv_exp_modules_bn_parameters_weight_ = l_self_modules_features_modules_4_modules_conv_exp_modules_bn_parameters_bias_ = (None)
        x_36 = torch.nn.functional.silu(x_35, inplace=True)
        x_35 = None
        x_37 = torch.conv2d(
            x_36,
            l_self_modules_features_modules_4_modules_conv_dw_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            450,
        )
        x_36 = l_self_modules_features_modules_4_modules_conv_dw_modules_conv_parameters_weight_ = (None)
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_features_modules_4_modules_conv_dw_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_conv_dw_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_4_modules_conv_dw_modules_bn_parameters_weight_,
            l_self_modules_features_modules_4_modules_conv_dw_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_37 = l_self_modules_features_modules_4_modules_conv_dw_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_4_modules_conv_dw_modules_bn_buffers_running_var_ = l_self_modules_features_modules_4_modules_conv_dw_modules_bn_parameters_weight_ = l_self_modules_features_modules_4_modules_conv_dw_modules_bn_parameters_bias_ = (None)
        x_se_4 = x_38.mean((2, 3), keepdim=True)
        x_se_5 = torch.conv2d(
            x_se_4,
            l_self_modules_features_modules_4_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_4 = (
            l_self_modules_features_modules_4_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_features_modules_4_modules_se_modules_fc1_parameters_bias_
        ) = None
        batch_norm_15 = torch.nn.functional.batch_norm(
            x_se_5,
            l_self_modules_features_modules_4_modules_se_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_se_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_4_modules_se_modules_bn_parameters_weight_,
            l_self_modules_features_modules_4_modules_se_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_se_5 = l_self_modules_features_modules_4_modules_se_modules_bn_buffers_running_mean_ = (
            l_self_modules_features_modules_4_modules_se_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_features_modules_4_modules_se_modules_bn_parameters_weight_
        ) = (
            l_self_modules_features_modules_4_modules_se_modules_bn_parameters_bias_
        ) = None
        x_se_6 = torch.nn.functional.relu(batch_norm_15, inplace=True)
        batch_norm_15 = None
        x_se_7 = torch.conv2d(
            x_se_6,
            l_self_modules_features_modules_4_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_6 = (
            l_self_modules_features_modules_4_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_features_modules_4_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_1 = x_se_7.sigmoid()
        x_se_7 = None
        x_39 = x_38 * sigmoid_1
        x_38 = sigmoid_1 = None
        x_40 = torch.nn.functional.hardtanh(x_39, 0.0, 6.0, False)
        x_39 = None
        x_41 = torch.conv2d(
            x_40,
            l_self_modules_features_modules_4_modules_conv_pwl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_40 = l_self_modules_features_modules_4_modules_conv_pwl_modules_conv_parameters_weight_ = (None)
        x_42 = torch.nn.functional.batch_norm(
            x_41,
            l_self_modules_features_modules_4_modules_conv_pwl_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_conv_pwl_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_4_modules_conv_pwl_modules_bn_parameters_weight_,
            l_self_modules_features_modules_4_modules_conv_pwl_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_41 = l_self_modules_features_modules_4_modules_conv_pwl_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_4_modules_conv_pwl_modules_bn_buffers_running_var_ = l_self_modules_features_modules_4_modules_conv_pwl_modules_bn_parameters_weight_ = l_self_modules_features_modules_4_modules_conv_pwl_modules_bn_parameters_bias_ = (None)
        getitem_2 = x_42[(slice(None, None, None), slice(0, 75, None))]
        add_1 = getitem_2 + x_33
        getitem_2 = x_33 = None
        getitem_3 = x_42[(slice(None, None, None), slice(75, None, None))]
        x_42 = None
        x_43 = torch.cat([add_1, getitem_3], dim=1)
        add_1 = getitem_3 = None
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_features_modules_5_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_43 = l_self_modules_features_modules_5_modules_conv_exp_modules_conv_parameters_weight_ = (None)
        x_45 = torch.nn.functional.batch_norm(
            x_44,
            l_self_modules_features_modules_5_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_5_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_features_modules_5_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_44 = l_self_modules_features_modules_5_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_5_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_features_modules_5_modules_conv_exp_modules_bn_parameters_weight_ = l_self_modules_features_modules_5_modules_conv_exp_modules_bn_parameters_bias_ = (None)
        x_46 = torch.nn.functional.silu(x_45, inplace=True)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_features_modules_5_modules_conv_dw_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            552,
        )
        x_46 = l_self_modules_features_modules_5_modules_conv_dw_modules_conv_parameters_weight_ = (None)
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_features_modules_5_modules_conv_dw_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_conv_dw_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_5_modules_conv_dw_modules_bn_parameters_weight_,
            l_self_modules_features_modules_5_modules_conv_dw_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = l_self_modules_features_modules_5_modules_conv_dw_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_5_modules_conv_dw_modules_bn_buffers_running_var_ = l_self_modules_features_modules_5_modules_conv_dw_modules_bn_parameters_weight_ = l_self_modules_features_modules_5_modules_conv_dw_modules_bn_parameters_bias_ = (None)
        x_se_8 = x_48.mean((2, 3), keepdim=True)
        x_se_9 = torch.conv2d(
            x_se_8,
            l_self_modules_features_modules_5_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_8 = (
            l_self_modules_features_modules_5_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_se_modules_fc1_parameters_bias_
        ) = None
        batch_norm_19 = torch.nn.functional.batch_norm(
            x_se_9,
            l_self_modules_features_modules_5_modules_se_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_se_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_5_modules_se_modules_bn_parameters_weight_,
            l_self_modules_features_modules_5_modules_se_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_se_9 = l_self_modules_features_modules_5_modules_se_modules_bn_buffers_running_mean_ = (
            l_self_modules_features_modules_5_modules_se_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_features_modules_5_modules_se_modules_bn_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_se_modules_bn_parameters_bias_
        ) = None
        x_se_10 = torch.nn.functional.relu(batch_norm_19, inplace=True)
        batch_norm_19 = None
        x_se_11 = torch.conv2d(
            x_se_10,
            l_self_modules_features_modules_5_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_10 = (
            l_self_modules_features_modules_5_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_2 = x_se_11.sigmoid()
        x_se_11 = None
        x_49 = x_48 * sigmoid_2
        x_48 = sigmoid_2 = None
        x_50 = torch.nn.functional.hardtanh(x_49, 0.0, 6.0, False)
        x_49 = None
        x_51 = torch.conv2d(
            x_50,
            l_self_modules_features_modules_5_modules_conv_pwl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_50 = l_self_modules_features_modules_5_modules_conv_pwl_modules_conv_parameters_weight_ = (None)
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_features_modules_5_modules_conv_pwl_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_conv_pwl_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_5_modules_conv_pwl_modules_bn_parameters_weight_,
            l_self_modules_features_modules_5_modules_conv_pwl_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_51 = l_self_modules_features_modules_5_modules_conv_pwl_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_5_modules_conv_pwl_modules_bn_buffers_running_var_ = l_self_modules_features_modules_5_modules_conv_pwl_modules_bn_parameters_weight_ = l_self_modules_features_modules_5_modules_conv_pwl_modules_bn_parameters_bias_ = (None)
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_features_modules_6_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_conv_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_features_modules_6_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_6_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_features_modules_6_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = l_self_modules_features_modules_6_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_6_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_features_modules_6_modules_conv_exp_modules_bn_parameters_weight_ = l_self_modules_features_modules_6_modules_conv_exp_modules_bn_parameters_bias_ = (None)
        x_55 = torch.nn.functional.silu(x_54, inplace=True)
        x_54 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_features_modules_6_modules_conv_dw_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            648,
        )
        x_55 = l_self_modules_features_modules_6_modules_conv_dw_modules_conv_parameters_weight_ = (None)
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_features_modules_6_modules_conv_dw_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_conv_dw_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_6_modules_conv_dw_modules_bn_parameters_weight_,
            l_self_modules_features_modules_6_modules_conv_dw_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_features_modules_6_modules_conv_dw_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_6_modules_conv_dw_modules_bn_buffers_running_var_ = l_self_modules_features_modules_6_modules_conv_dw_modules_bn_parameters_weight_ = l_self_modules_features_modules_6_modules_conv_dw_modules_bn_parameters_bias_ = (None)
        x_se_12 = x_57.mean((2, 3), keepdim=True)
        x_se_13 = torch.conv2d(
            x_se_12,
            l_self_modules_features_modules_6_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_12 = (
            l_self_modules_features_modules_6_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_features_modules_6_modules_se_modules_fc1_parameters_bias_
        ) = None
        batch_norm_23 = torch.nn.functional.batch_norm(
            x_se_13,
            l_self_modules_features_modules_6_modules_se_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_se_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_6_modules_se_modules_bn_parameters_weight_,
            l_self_modules_features_modules_6_modules_se_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_se_13 = l_self_modules_features_modules_6_modules_se_modules_bn_buffers_running_mean_ = (
            l_self_modules_features_modules_6_modules_se_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_features_modules_6_modules_se_modules_bn_parameters_weight_
        ) = (
            l_self_modules_features_modules_6_modules_se_modules_bn_parameters_bias_
        ) = None
        x_se_14 = torch.nn.functional.relu(batch_norm_23, inplace=True)
        batch_norm_23 = None
        x_se_15 = torch.conv2d(
            x_se_14,
            l_self_modules_features_modules_6_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_14 = (
            l_self_modules_features_modules_6_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_features_modules_6_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_3 = x_se_15.sigmoid()
        x_se_15 = None
        x_58 = x_57 * sigmoid_3
        x_57 = sigmoid_3 = None
        x_59 = torch.nn.functional.hardtanh(x_58, 0.0, 6.0, False)
        x_58 = None
        x_60 = torch.conv2d(
            x_59,
            l_self_modules_features_modules_6_modules_conv_pwl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_59 = l_self_modules_features_modules_6_modules_conv_pwl_modules_conv_parameters_weight_ = (None)
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_features_modules_6_modules_conv_pwl_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_conv_pwl_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_6_modules_conv_pwl_modules_bn_parameters_weight_,
            l_self_modules_features_modules_6_modules_conv_pwl_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_60 = l_self_modules_features_modules_6_modules_conv_pwl_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_6_modules_conv_pwl_modules_bn_buffers_running_var_ = l_self_modules_features_modules_6_modules_conv_pwl_modules_bn_parameters_weight_ = l_self_modules_features_modules_6_modules_conv_pwl_modules_bn_parameters_bias_ = (None)
        getitem_4 = x_61[(slice(None, None, None), slice(0, 108, None))]
        add_2 = getitem_4 + x_52
        getitem_4 = x_52 = None
        getitem_5 = x_61[(slice(None, None, None), slice(108, None, None))]
        x_61 = None
        x_62 = torch.cat([add_2, getitem_5], dim=1)
        add_2 = getitem_5 = None
        x_63 = torch.conv2d(
            x_62,
            l_self_modules_features_modules_7_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_7_modules_conv_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_features_modules_7_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_7_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_features_modules_7_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = l_self_modules_features_modules_7_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_7_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_features_modules_7_modules_conv_exp_modules_bn_parameters_weight_ = l_self_modules_features_modules_7_modules_conv_exp_modules_bn_parameters_bias_ = (None)
        x_65 = torch.nn.functional.silu(x_64, inplace=True)
        x_64 = None
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_features_modules_7_modules_conv_dw_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            750,
        )
        x_65 = l_self_modules_features_modules_7_modules_conv_dw_modules_conv_parameters_weight_ = (None)
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_features_modules_7_modules_conv_dw_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_conv_dw_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_7_modules_conv_dw_modules_bn_parameters_weight_,
            l_self_modules_features_modules_7_modules_conv_dw_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = l_self_modules_features_modules_7_modules_conv_dw_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_7_modules_conv_dw_modules_bn_buffers_running_var_ = l_self_modules_features_modules_7_modules_conv_dw_modules_bn_parameters_weight_ = l_self_modules_features_modules_7_modules_conv_dw_modules_bn_parameters_bias_ = (None)
        x_se_16 = x_67.mean((2, 3), keepdim=True)
        x_se_17 = torch.conv2d(
            x_se_16,
            l_self_modules_features_modules_7_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_16 = (
            l_self_modules_features_modules_7_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_features_modules_7_modules_se_modules_fc1_parameters_bias_
        ) = None
        batch_norm_27 = torch.nn.functional.batch_norm(
            x_se_17,
            l_self_modules_features_modules_7_modules_se_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_se_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_7_modules_se_modules_bn_parameters_weight_,
            l_self_modules_features_modules_7_modules_se_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_se_17 = l_self_modules_features_modules_7_modules_se_modules_bn_buffers_running_mean_ = (
            l_self_modules_features_modules_7_modules_se_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_features_modules_7_modules_se_modules_bn_parameters_weight_
        ) = (
            l_self_modules_features_modules_7_modules_se_modules_bn_parameters_bias_
        ) = None
        x_se_18 = torch.nn.functional.relu(batch_norm_27, inplace=True)
        batch_norm_27 = None
        x_se_19 = torch.conv2d(
            x_se_18,
            l_self_modules_features_modules_7_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_18 = (
            l_self_modules_features_modules_7_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_features_modules_7_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_4 = x_se_19.sigmoid()
        x_se_19 = None
        x_68 = x_67 * sigmoid_4
        x_67 = sigmoid_4 = None
        x_69 = torch.nn.functional.hardtanh(x_68, 0.0, 6.0, False)
        x_68 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_features_modules_7_modules_conv_pwl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_69 = l_self_modules_features_modules_7_modules_conv_pwl_modules_conv_parameters_weight_ = (None)
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_features_modules_7_modules_conv_pwl_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_conv_pwl_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_7_modules_conv_pwl_modules_bn_parameters_weight_,
            l_self_modules_features_modules_7_modules_conv_pwl_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = l_self_modules_features_modules_7_modules_conv_pwl_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_7_modules_conv_pwl_modules_bn_buffers_running_var_ = l_self_modules_features_modules_7_modules_conv_pwl_modules_bn_parameters_weight_ = l_self_modules_features_modules_7_modules_conv_pwl_modules_bn_parameters_bias_ = (None)
        getitem_6 = x_71[(slice(None, None, None), slice(0, 125, None))]
        add_3 = getitem_6 + x_62
        getitem_6 = x_62 = None
        getitem_7 = x_71[(slice(None, None, None), slice(125, None, None))]
        x_71 = None
        x_72 = torch.cat([add_3, getitem_7], dim=1)
        add_3 = getitem_7 = None
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_features_modules_8_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_8_modules_conv_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_features_modules_8_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_8_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_features_modules_8_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = l_self_modules_features_modules_8_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_8_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_features_modules_8_modules_conv_exp_modules_bn_parameters_weight_ = l_self_modules_features_modules_8_modules_conv_exp_modules_bn_parameters_bias_ = (None)
        x_75 = torch.nn.functional.silu(x_74, inplace=True)
        x_74 = None
        x_76 = torch.conv2d(
            x_75,
            l_self_modules_features_modules_8_modules_conv_dw_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            852,
        )
        x_75 = l_self_modules_features_modules_8_modules_conv_dw_modules_conv_parameters_weight_ = (None)
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_features_modules_8_modules_conv_dw_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_conv_dw_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_8_modules_conv_dw_modules_bn_parameters_weight_,
            l_self_modules_features_modules_8_modules_conv_dw_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = l_self_modules_features_modules_8_modules_conv_dw_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_8_modules_conv_dw_modules_bn_buffers_running_var_ = l_self_modules_features_modules_8_modules_conv_dw_modules_bn_parameters_weight_ = l_self_modules_features_modules_8_modules_conv_dw_modules_bn_parameters_bias_ = (None)
        x_se_20 = x_77.mean((2, 3), keepdim=True)
        x_se_21 = torch.conv2d(
            x_se_20,
            l_self_modules_features_modules_8_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_8_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_20 = (
            l_self_modules_features_modules_8_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_features_modules_8_modules_se_modules_fc1_parameters_bias_
        ) = None
        batch_norm_31 = torch.nn.functional.batch_norm(
            x_se_21,
            l_self_modules_features_modules_8_modules_se_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_se_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_8_modules_se_modules_bn_parameters_weight_,
            l_self_modules_features_modules_8_modules_se_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_se_21 = l_self_modules_features_modules_8_modules_se_modules_bn_buffers_running_mean_ = (
            l_self_modules_features_modules_8_modules_se_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_features_modules_8_modules_se_modules_bn_parameters_weight_
        ) = (
            l_self_modules_features_modules_8_modules_se_modules_bn_parameters_bias_
        ) = None
        x_se_22 = torch.nn.functional.relu(batch_norm_31, inplace=True)
        batch_norm_31 = None
        x_se_23 = torch.conv2d(
            x_se_22,
            l_self_modules_features_modules_8_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_8_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_22 = (
            l_self_modules_features_modules_8_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_features_modules_8_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_5 = x_se_23.sigmoid()
        x_se_23 = None
        x_78 = x_77 * sigmoid_5
        x_77 = sigmoid_5 = None
        x_79 = torch.nn.functional.hardtanh(x_78, 0.0, 6.0, False)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_features_modules_8_modules_conv_pwl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_features_modules_8_modules_conv_pwl_modules_conv_parameters_weight_ = (None)
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_features_modules_8_modules_conv_pwl_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_conv_pwl_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_8_modules_conv_pwl_modules_bn_parameters_weight_,
            l_self_modules_features_modules_8_modules_conv_pwl_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = l_self_modules_features_modules_8_modules_conv_pwl_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_8_modules_conv_pwl_modules_bn_buffers_running_var_ = l_self_modules_features_modules_8_modules_conv_pwl_modules_bn_parameters_weight_ = l_self_modules_features_modules_8_modules_conv_pwl_modules_bn_parameters_bias_ = (None)
        getitem_8 = x_81[(slice(None, None, None), slice(0, 142, None))]
        add_4 = getitem_8 + x_72
        getitem_8 = x_72 = None
        getitem_9 = x_81[(slice(None, None, None), slice(142, None, None))]
        x_81 = None
        x_82 = torch.cat([add_4, getitem_9], dim=1)
        add_4 = getitem_9 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_features_modules_9_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_9_modules_conv_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_features_modules_9_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_9_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_9_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_features_modules_9_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = l_self_modules_features_modules_9_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_9_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_features_modules_9_modules_conv_exp_modules_bn_parameters_weight_ = l_self_modules_features_modules_9_modules_conv_exp_modules_bn_parameters_bias_ = (None)
        x_85 = torch.nn.functional.silu(x_84, inplace=True)
        x_84 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_features_modules_9_modules_conv_dw_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            954,
        )
        x_85 = l_self_modules_features_modules_9_modules_conv_dw_modules_conv_parameters_weight_ = (None)
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_features_modules_9_modules_conv_dw_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_9_modules_conv_dw_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_9_modules_conv_dw_modules_bn_parameters_weight_,
            l_self_modules_features_modules_9_modules_conv_dw_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = l_self_modules_features_modules_9_modules_conv_dw_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_9_modules_conv_dw_modules_bn_buffers_running_var_ = l_self_modules_features_modules_9_modules_conv_dw_modules_bn_parameters_weight_ = l_self_modules_features_modules_9_modules_conv_dw_modules_bn_parameters_bias_ = (None)
        x_se_24 = x_87.mean((2, 3), keepdim=True)
        x_se_25 = torch.conv2d(
            x_se_24,
            l_self_modules_features_modules_9_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_9_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_24 = (
            l_self_modules_features_modules_9_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_features_modules_9_modules_se_modules_fc1_parameters_bias_
        ) = None
        batch_norm_35 = torch.nn.functional.batch_norm(
            x_se_25,
            l_self_modules_features_modules_9_modules_se_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_9_modules_se_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_9_modules_se_modules_bn_parameters_weight_,
            l_self_modules_features_modules_9_modules_se_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_se_25 = l_self_modules_features_modules_9_modules_se_modules_bn_buffers_running_mean_ = (
            l_self_modules_features_modules_9_modules_se_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_features_modules_9_modules_se_modules_bn_parameters_weight_
        ) = (
            l_self_modules_features_modules_9_modules_se_modules_bn_parameters_bias_
        ) = None
        x_se_26 = torch.nn.functional.relu(batch_norm_35, inplace=True)
        batch_norm_35 = None
        x_se_27 = torch.conv2d(
            x_se_26,
            l_self_modules_features_modules_9_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_9_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_26 = (
            l_self_modules_features_modules_9_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_features_modules_9_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_6 = x_se_27.sigmoid()
        x_se_27 = None
        x_88 = x_87 * sigmoid_6
        x_87 = sigmoid_6 = None
        x_89 = torch.nn.functional.hardtanh(x_88, 0.0, 6.0, False)
        x_88 = None
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_features_modules_9_modules_conv_pwl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_features_modules_9_modules_conv_pwl_modules_conv_parameters_weight_ = (None)
        x_91 = torch.nn.functional.batch_norm(
            x_90,
            l_self_modules_features_modules_9_modules_conv_pwl_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_9_modules_conv_pwl_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_9_modules_conv_pwl_modules_bn_parameters_weight_,
            l_self_modules_features_modules_9_modules_conv_pwl_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_90 = l_self_modules_features_modules_9_modules_conv_pwl_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_9_modules_conv_pwl_modules_bn_buffers_running_var_ = l_self_modules_features_modules_9_modules_conv_pwl_modules_bn_parameters_weight_ = l_self_modules_features_modules_9_modules_conv_pwl_modules_bn_parameters_bias_ = (None)
        getitem_10 = x_91[(slice(None, None, None), slice(0, 159, None))]
        add_5 = getitem_10 + x_82
        getitem_10 = x_82 = None
        getitem_11 = x_91[(slice(None, None, None), slice(159, None, None))]
        x_91 = None
        x_92 = torch.cat([add_5, getitem_11], dim=1)
        add_5 = getitem_11 = None
        x_93 = torch.conv2d(
            x_92,
            l_self_modules_features_modules_10_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_10_modules_conv_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_94 = torch.nn.functional.batch_norm(
            x_93,
            l_self_modules_features_modules_10_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_10_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_10_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_features_modules_10_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_93 = l_self_modules_features_modules_10_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_10_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_features_modules_10_modules_conv_exp_modules_bn_parameters_weight_ = l_self_modules_features_modules_10_modules_conv_exp_modules_bn_parameters_bias_ = (None)
        x_95 = torch.nn.functional.silu(x_94, inplace=True)
        x_94 = None
        x_96 = torch.conv2d(
            x_95,
            l_self_modules_features_modules_10_modules_conv_dw_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1056,
        )
        x_95 = l_self_modules_features_modules_10_modules_conv_dw_modules_conv_parameters_weight_ = (None)
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_features_modules_10_modules_conv_dw_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_10_modules_conv_dw_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_10_modules_conv_dw_modules_bn_parameters_weight_,
            l_self_modules_features_modules_10_modules_conv_dw_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_96 = l_self_modules_features_modules_10_modules_conv_dw_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_10_modules_conv_dw_modules_bn_buffers_running_var_ = l_self_modules_features_modules_10_modules_conv_dw_modules_bn_parameters_weight_ = l_self_modules_features_modules_10_modules_conv_dw_modules_bn_parameters_bias_ = (None)
        x_se_28 = x_97.mean((2, 3), keepdim=True)
        x_se_29 = torch.conv2d(
            x_se_28,
            l_self_modules_features_modules_10_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_10_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_28 = (
            l_self_modules_features_modules_10_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_features_modules_10_modules_se_modules_fc1_parameters_bias_
        ) = None
        batch_norm_39 = torch.nn.functional.batch_norm(
            x_se_29,
            l_self_modules_features_modules_10_modules_se_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_10_modules_se_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_10_modules_se_modules_bn_parameters_weight_,
            l_self_modules_features_modules_10_modules_se_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_se_29 = l_self_modules_features_modules_10_modules_se_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_10_modules_se_modules_bn_buffers_running_var_ = (
            l_self_modules_features_modules_10_modules_se_modules_bn_parameters_weight_
        ) = (
            l_self_modules_features_modules_10_modules_se_modules_bn_parameters_bias_
        ) = None
        x_se_30 = torch.nn.functional.relu(batch_norm_39, inplace=True)
        batch_norm_39 = None
        x_se_31 = torch.conv2d(
            x_se_30,
            l_self_modules_features_modules_10_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_10_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_30 = (
            l_self_modules_features_modules_10_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_features_modules_10_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_7 = x_se_31.sigmoid()
        x_se_31 = None
        x_98 = x_97 * sigmoid_7
        x_97 = sigmoid_7 = None
        x_99 = torch.nn.functional.hardtanh(x_98, 0.0, 6.0, False)
        x_98 = None
        x_100 = torch.conv2d(
            x_99,
            l_self_modules_features_modules_10_modules_conv_pwl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_99 = l_self_modules_features_modules_10_modules_conv_pwl_modules_conv_parameters_weight_ = (None)
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_features_modules_10_modules_conv_pwl_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_10_modules_conv_pwl_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_10_modules_conv_pwl_modules_bn_parameters_weight_,
            l_self_modules_features_modules_10_modules_conv_pwl_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_100 = l_self_modules_features_modules_10_modules_conv_pwl_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_10_modules_conv_pwl_modules_bn_buffers_running_var_ = l_self_modules_features_modules_10_modules_conv_pwl_modules_bn_parameters_weight_ = l_self_modules_features_modules_10_modules_conv_pwl_modules_bn_parameters_bias_ = (None)
        getitem_12 = x_101[(slice(None, None, None), slice(0, 176, None))]
        add_6 = getitem_12 + x_92
        getitem_12 = x_92 = None
        getitem_13 = x_101[(slice(None, None, None), slice(176, None, None))]
        x_101 = None
        x_102 = torch.cat([add_6, getitem_13], dim=1)
        add_6 = getitem_13 = None
        x_103 = torch.conv2d(
            x_102,
            l_self_modules_features_modules_11_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_102 = l_self_modules_features_modules_11_modules_conv_exp_modules_conv_parameters_weight_ = (None)
        x_104 = torch.nn.functional.batch_norm(
            x_103,
            l_self_modules_features_modules_11_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_11_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_features_modules_11_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_103 = l_self_modules_features_modules_11_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_11_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_features_modules_11_modules_conv_exp_modules_bn_parameters_weight_ = l_self_modules_features_modules_11_modules_conv_exp_modules_bn_parameters_bias_ = (None)
        x_105 = torch.nn.functional.silu(x_104, inplace=True)
        x_104 = None
        x_106 = torch.conv2d(
            x_105,
            l_self_modules_features_modules_11_modules_conv_dw_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1158,
        )
        x_105 = l_self_modules_features_modules_11_modules_conv_dw_modules_conv_parameters_weight_ = (None)
        x_107 = torch.nn.functional.batch_norm(
            x_106,
            l_self_modules_features_modules_11_modules_conv_dw_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_conv_dw_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_11_modules_conv_dw_modules_bn_parameters_weight_,
            l_self_modules_features_modules_11_modules_conv_dw_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_106 = l_self_modules_features_modules_11_modules_conv_dw_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_11_modules_conv_dw_modules_bn_buffers_running_var_ = l_self_modules_features_modules_11_modules_conv_dw_modules_bn_parameters_weight_ = l_self_modules_features_modules_11_modules_conv_dw_modules_bn_parameters_bias_ = (None)
        x_se_32 = x_107.mean((2, 3), keepdim=True)
        x_se_33 = torch.conv2d(
            x_se_32,
            l_self_modules_features_modules_11_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_11_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_32 = (
            l_self_modules_features_modules_11_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_features_modules_11_modules_se_modules_fc1_parameters_bias_
        ) = None
        batch_norm_43 = torch.nn.functional.batch_norm(
            x_se_33,
            l_self_modules_features_modules_11_modules_se_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_se_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_11_modules_se_modules_bn_parameters_weight_,
            l_self_modules_features_modules_11_modules_se_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_se_33 = l_self_modules_features_modules_11_modules_se_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_11_modules_se_modules_bn_buffers_running_var_ = (
            l_self_modules_features_modules_11_modules_se_modules_bn_parameters_weight_
        ) = (
            l_self_modules_features_modules_11_modules_se_modules_bn_parameters_bias_
        ) = None
        x_se_34 = torch.nn.functional.relu(batch_norm_43, inplace=True)
        batch_norm_43 = None
        x_se_35 = torch.conv2d(
            x_se_34,
            l_self_modules_features_modules_11_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_11_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_34 = (
            l_self_modules_features_modules_11_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_features_modules_11_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_8 = x_se_35.sigmoid()
        x_se_35 = None
        x_108 = x_107 * sigmoid_8
        x_107 = sigmoid_8 = None
        x_109 = torch.nn.functional.hardtanh(x_108, 0.0, 6.0, False)
        x_108 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_features_modules_11_modules_conv_pwl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_features_modules_11_modules_conv_pwl_modules_conv_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_features_modules_11_modules_conv_pwl_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_conv_pwl_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_11_modules_conv_pwl_modules_bn_parameters_weight_,
            l_self_modules_features_modules_11_modules_conv_pwl_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_features_modules_11_modules_conv_pwl_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_11_modules_conv_pwl_modules_bn_buffers_running_var_ = l_self_modules_features_modules_11_modules_conv_pwl_modules_bn_parameters_weight_ = l_self_modules_features_modules_11_modules_conv_pwl_modules_bn_parameters_bias_ = (None)
        x_112 = torch.conv2d(
            x_111,
            l_self_modules_features_modules_12_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_12_modules_conv_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_features_modules_12_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_12_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_12_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_features_modules_12_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_112 = l_self_modules_features_modules_12_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_12_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_features_modules_12_modules_conv_exp_modules_bn_parameters_weight_ = l_self_modules_features_modules_12_modules_conv_exp_modules_bn_parameters_bias_ = (None)
        x_114 = torch.nn.functional.silu(x_113, inplace=True)
        x_113 = None
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_features_modules_12_modules_conv_dw_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1260,
        )
        x_114 = l_self_modules_features_modules_12_modules_conv_dw_modules_conv_parameters_weight_ = (None)
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_features_modules_12_modules_conv_dw_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_12_modules_conv_dw_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_12_modules_conv_dw_modules_bn_parameters_weight_,
            l_self_modules_features_modules_12_modules_conv_dw_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_115 = l_self_modules_features_modules_12_modules_conv_dw_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_12_modules_conv_dw_modules_bn_buffers_running_var_ = l_self_modules_features_modules_12_modules_conv_dw_modules_bn_parameters_weight_ = l_self_modules_features_modules_12_modules_conv_dw_modules_bn_parameters_bias_ = (None)
        x_se_36 = x_116.mean((2, 3), keepdim=True)
        x_se_37 = torch.conv2d(
            x_se_36,
            l_self_modules_features_modules_12_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_12_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_36 = (
            l_self_modules_features_modules_12_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_features_modules_12_modules_se_modules_fc1_parameters_bias_
        ) = None
        batch_norm_47 = torch.nn.functional.batch_norm(
            x_se_37,
            l_self_modules_features_modules_12_modules_se_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_12_modules_se_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_12_modules_se_modules_bn_parameters_weight_,
            l_self_modules_features_modules_12_modules_se_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_se_37 = l_self_modules_features_modules_12_modules_se_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_12_modules_se_modules_bn_buffers_running_var_ = (
            l_self_modules_features_modules_12_modules_se_modules_bn_parameters_weight_
        ) = (
            l_self_modules_features_modules_12_modules_se_modules_bn_parameters_bias_
        ) = None
        x_se_38 = torch.nn.functional.relu(batch_norm_47, inplace=True)
        batch_norm_47 = None
        x_se_39 = torch.conv2d(
            x_se_38,
            l_self_modules_features_modules_12_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_12_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_38 = (
            l_self_modules_features_modules_12_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_features_modules_12_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_9 = x_se_39.sigmoid()
        x_se_39 = None
        x_117 = x_116 * sigmoid_9
        x_116 = sigmoid_9 = None
        x_118 = torch.nn.functional.hardtanh(x_117, 0.0, 6.0, False)
        x_117 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_features_modules_12_modules_conv_pwl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_features_modules_12_modules_conv_pwl_modules_conv_parameters_weight_ = (None)
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_features_modules_12_modules_conv_pwl_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_12_modules_conv_pwl_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_12_modules_conv_pwl_modules_bn_parameters_weight_,
            l_self_modules_features_modules_12_modules_conv_pwl_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = l_self_modules_features_modules_12_modules_conv_pwl_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_12_modules_conv_pwl_modules_bn_buffers_running_var_ = l_self_modules_features_modules_12_modules_conv_pwl_modules_bn_parameters_weight_ = l_self_modules_features_modules_12_modules_conv_pwl_modules_bn_parameters_bias_ = (None)
        getitem_14 = x_120[(slice(None, None, None), slice(0, 210, None))]
        add_7 = getitem_14 + x_111
        getitem_14 = x_111 = None
        getitem_15 = x_120[(slice(None, None, None), slice(210, None, None))]
        x_120 = None
        x_121 = torch.cat([add_7, getitem_15], dim=1)
        add_7 = getitem_15 = None
        x_122 = torch.conv2d(
            x_121,
            l_self_modules_features_modules_13_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_13_modules_conv_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_123 = torch.nn.functional.batch_norm(
            x_122,
            l_self_modules_features_modules_13_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_13_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_13_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_features_modules_13_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_122 = l_self_modules_features_modules_13_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_13_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_features_modules_13_modules_conv_exp_modules_bn_parameters_weight_ = l_self_modules_features_modules_13_modules_conv_exp_modules_bn_parameters_bias_ = (None)
        x_124 = torch.nn.functional.silu(x_123, inplace=True)
        x_123 = None
        x_125 = torch.conv2d(
            x_124,
            l_self_modules_features_modules_13_modules_conv_dw_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1356,
        )
        x_124 = l_self_modules_features_modules_13_modules_conv_dw_modules_conv_parameters_weight_ = (None)
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_self_modules_features_modules_13_modules_conv_dw_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_13_modules_conv_dw_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_13_modules_conv_dw_modules_bn_parameters_weight_,
            l_self_modules_features_modules_13_modules_conv_dw_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_125 = l_self_modules_features_modules_13_modules_conv_dw_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_13_modules_conv_dw_modules_bn_buffers_running_var_ = l_self_modules_features_modules_13_modules_conv_dw_modules_bn_parameters_weight_ = l_self_modules_features_modules_13_modules_conv_dw_modules_bn_parameters_bias_ = (None)
        x_se_40 = x_126.mean((2, 3), keepdim=True)
        x_se_41 = torch.conv2d(
            x_se_40,
            l_self_modules_features_modules_13_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_13_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_40 = (
            l_self_modules_features_modules_13_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_features_modules_13_modules_se_modules_fc1_parameters_bias_
        ) = None
        batch_norm_51 = torch.nn.functional.batch_norm(
            x_se_41,
            l_self_modules_features_modules_13_modules_se_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_13_modules_se_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_13_modules_se_modules_bn_parameters_weight_,
            l_self_modules_features_modules_13_modules_se_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_se_41 = l_self_modules_features_modules_13_modules_se_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_13_modules_se_modules_bn_buffers_running_var_ = (
            l_self_modules_features_modules_13_modules_se_modules_bn_parameters_weight_
        ) = (
            l_self_modules_features_modules_13_modules_se_modules_bn_parameters_bias_
        ) = None
        x_se_42 = torch.nn.functional.relu(batch_norm_51, inplace=True)
        batch_norm_51 = None
        x_se_43 = torch.conv2d(
            x_se_42,
            l_self_modules_features_modules_13_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_13_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_42 = (
            l_self_modules_features_modules_13_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_features_modules_13_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_10 = x_se_43.sigmoid()
        x_se_43 = None
        x_127 = x_126 * sigmoid_10
        x_126 = sigmoid_10 = None
        x_128 = torch.nn.functional.hardtanh(x_127, 0.0, 6.0, False)
        x_127 = None
        x_129 = torch.conv2d(
            x_128,
            l_self_modules_features_modules_13_modules_conv_pwl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_128 = l_self_modules_features_modules_13_modules_conv_pwl_modules_conv_parameters_weight_ = (None)
        x_130 = torch.nn.functional.batch_norm(
            x_129,
            l_self_modules_features_modules_13_modules_conv_pwl_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_13_modules_conv_pwl_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_13_modules_conv_pwl_modules_bn_parameters_weight_,
            l_self_modules_features_modules_13_modules_conv_pwl_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_129 = l_self_modules_features_modules_13_modules_conv_pwl_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_13_modules_conv_pwl_modules_bn_buffers_running_var_ = l_self_modules_features_modules_13_modules_conv_pwl_modules_bn_parameters_weight_ = l_self_modules_features_modules_13_modules_conv_pwl_modules_bn_parameters_bias_ = (None)
        getitem_16 = x_130[(slice(None, None, None), slice(0, 226, None))]
        add_8 = getitem_16 + x_121
        getitem_16 = x_121 = None
        getitem_17 = x_130[(slice(None, None, None), slice(226, None, None))]
        x_130 = None
        x_131 = torch.cat([add_8, getitem_17], dim=1)
        add_8 = getitem_17 = None
        x_132 = torch.conv2d(
            x_131,
            l_self_modules_features_modules_14_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_14_modules_conv_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_133 = torch.nn.functional.batch_norm(
            x_132,
            l_self_modules_features_modules_14_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_14_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_14_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_features_modules_14_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_132 = l_self_modules_features_modules_14_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_14_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_features_modules_14_modules_conv_exp_modules_bn_parameters_weight_ = l_self_modules_features_modules_14_modules_conv_exp_modules_bn_parameters_bias_ = (None)
        x_134 = torch.nn.functional.silu(x_133, inplace=True)
        x_133 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_features_modules_14_modules_conv_dw_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1458,
        )
        x_134 = l_self_modules_features_modules_14_modules_conv_dw_modules_conv_parameters_weight_ = (None)
        x_136 = torch.nn.functional.batch_norm(
            x_135,
            l_self_modules_features_modules_14_modules_conv_dw_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_14_modules_conv_dw_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_14_modules_conv_dw_modules_bn_parameters_weight_,
            l_self_modules_features_modules_14_modules_conv_dw_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_135 = l_self_modules_features_modules_14_modules_conv_dw_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_14_modules_conv_dw_modules_bn_buffers_running_var_ = l_self_modules_features_modules_14_modules_conv_dw_modules_bn_parameters_weight_ = l_self_modules_features_modules_14_modules_conv_dw_modules_bn_parameters_bias_ = (None)
        x_se_44 = x_136.mean((2, 3), keepdim=True)
        x_se_45 = torch.conv2d(
            x_se_44,
            l_self_modules_features_modules_14_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_14_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_44 = (
            l_self_modules_features_modules_14_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_features_modules_14_modules_se_modules_fc1_parameters_bias_
        ) = None
        batch_norm_55 = torch.nn.functional.batch_norm(
            x_se_45,
            l_self_modules_features_modules_14_modules_se_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_14_modules_se_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_14_modules_se_modules_bn_parameters_weight_,
            l_self_modules_features_modules_14_modules_se_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_se_45 = l_self_modules_features_modules_14_modules_se_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_14_modules_se_modules_bn_buffers_running_var_ = (
            l_self_modules_features_modules_14_modules_se_modules_bn_parameters_weight_
        ) = (
            l_self_modules_features_modules_14_modules_se_modules_bn_parameters_bias_
        ) = None
        x_se_46 = torch.nn.functional.relu(batch_norm_55, inplace=True)
        batch_norm_55 = None
        x_se_47 = torch.conv2d(
            x_se_46,
            l_self_modules_features_modules_14_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_14_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_46 = (
            l_self_modules_features_modules_14_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_features_modules_14_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_11 = x_se_47.sigmoid()
        x_se_47 = None
        x_137 = x_136 * sigmoid_11
        x_136 = sigmoid_11 = None
        x_138 = torch.nn.functional.hardtanh(x_137, 0.0, 6.0, False)
        x_137 = None
        x_139 = torch.conv2d(
            x_138,
            l_self_modules_features_modules_14_modules_conv_pwl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_138 = l_self_modules_features_modules_14_modules_conv_pwl_modules_conv_parameters_weight_ = (None)
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_features_modules_14_modules_conv_pwl_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_14_modules_conv_pwl_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_14_modules_conv_pwl_modules_bn_parameters_weight_,
            l_self_modules_features_modules_14_modules_conv_pwl_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_139 = l_self_modules_features_modules_14_modules_conv_pwl_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_14_modules_conv_pwl_modules_bn_buffers_running_var_ = l_self_modules_features_modules_14_modules_conv_pwl_modules_bn_parameters_weight_ = l_self_modules_features_modules_14_modules_conv_pwl_modules_bn_parameters_bias_ = (None)
        getitem_18 = x_140[(slice(None, None, None), slice(0, 243, None))]
        add_9 = getitem_18 + x_131
        getitem_18 = x_131 = None
        getitem_19 = x_140[(slice(None, None, None), slice(243, None, None))]
        x_140 = None
        x_141 = torch.cat([add_9, getitem_19], dim=1)
        add_9 = getitem_19 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_features_modules_15_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_15_modules_conv_exp_modules_conv_parameters_weight_ = (
            None
        )
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_features_modules_15_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_15_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_15_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_features_modules_15_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_142 = l_self_modules_features_modules_15_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_15_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_features_modules_15_modules_conv_exp_modules_bn_parameters_weight_ = l_self_modules_features_modules_15_modules_conv_exp_modules_bn_parameters_bias_ = (None)
        x_144 = torch.nn.functional.silu(x_143, inplace=True)
        x_143 = None
        x_145 = torch.conv2d(
            x_144,
            l_self_modules_features_modules_15_modules_conv_dw_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1560,
        )
        x_144 = l_self_modules_features_modules_15_modules_conv_dw_modules_conv_parameters_weight_ = (None)
        x_146 = torch.nn.functional.batch_norm(
            x_145,
            l_self_modules_features_modules_15_modules_conv_dw_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_15_modules_conv_dw_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_15_modules_conv_dw_modules_bn_parameters_weight_,
            l_self_modules_features_modules_15_modules_conv_dw_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_145 = l_self_modules_features_modules_15_modules_conv_dw_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_15_modules_conv_dw_modules_bn_buffers_running_var_ = l_self_modules_features_modules_15_modules_conv_dw_modules_bn_parameters_weight_ = l_self_modules_features_modules_15_modules_conv_dw_modules_bn_parameters_bias_ = (None)
        x_se_48 = x_146.mean((2, 3), keepdim=True)
        x_se_49 = torch.conv2d(
            x_se_48,
            l_self_modules_features_modules_15_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_15_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_48 = (
            l_self_modules_features_modules_15_modules_se_modules_fc1_parameters_weight_
        ) = (
            l_self_modules_features_modules_15_modules_se_modules_fc1_parameters_bias_
        ) = None
        batch_norm_59 = torch.nn.functional.batch_norm(
            x_se_49,
            l_self_modules_features_modules_15_modules_se_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_15_modules_se_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_15_modules_se_modules_bn_parameters_weight_,
            l_self_modules_features_modules_15_modules_se_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_se_49 = l_self_modules_features_modules_15_modules_se_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_15_modules_se_modules_bn_buffers_running_var_ = (
            l_self_modules_features_modules_15_modules_se_modules_bn_parameters_weight_
        ) = (
            l_self_modules_features_modules_15_modules_se_modules_bn_parameters_bias_
        ) = None
        x_se_50 = torch.nn.functional.relu(batch_norm_59, inplace=True)
        batch_norm_59 = None
        x_se_51 = torch.conv2d(
            x_se_50,
            l_self_modules_features_modules_15_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_15_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_50 = (
            l_self_modules_features_modules_15_modules_se_modules_fc2_parameters_weight_
        ) = (
            l_self_modules_features_modules_15_modules_se_modules_fc2_parameters_bias_
        ) = None
        sigmoid_12 = x_se_51.sigmoid()
        x_se_51 = None
        x_147 = x_146 * sigmoid_12
        x_146 = sigmoid_12 = None
        x_148 = torch.nn.functional.hardtanh(x_147, 0.0, 6.0, False)
        x_147 = None
        x_149 = torch.conv2d(
            x_148,
            l_self_modules_features_modules_15_modules_conv_pwl_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_148 = l_self_modules_features_modules_15_modules_conv_pwl_modules_conv_parameters_weight_ = (None)
        x_150 = torch.nn.functional.batch_norm(
            x_149,
            l_self_modules_features_modules_15_modules_conv_pwl_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_15_modules_conv_pwl_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_15_modules_conv_pwl_modules_bn_parameters_weight_,
            l_self_modules_features_modules_15_modules_conv_pwl_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_149 = l_self_modules_features_modules_15_modules_conv_pwl_modules_bn_buffers_running_mean_ = l_self_modules_features_modules_15_modules_conv_pwl_modules_bn_buffers_running_var_ = l_self_modules_features_modules_15_modules_conv_pwl_modules_bn_parameters_weight_ = l_self_modules_features_modules_15_modules_conv_pwl_modules_bn_parameters_bias_ = (None)
        getitem_20 = x_150[(slice(None, None, None), slice(0, 260, None))]
        add_10 = getitem_20 + x_141
        getitem_20 = x_141 = None
        getitem_21 = x_150[(slice(None, None, None), slice(260, None, None))]
        x_150 = None
        x_151 = torch.cat([add_10, getitem_21], dim=1)
        add_10 = getitem_21 = None
        x_152 = torch.conv2d(
            x_151,
            l_self_modules_features_modules_16_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_151 = (
            l_self_modules_features_modules_16_modules_conv_parameters_weight_
        ) = None
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_features_modules_16_modules_bn_buffers_running_mean_,
            l_self_modules_features_modules_16_modules_bn_buffers_running_var_,
            l_self_modules_features_modules_16_modules_bn_parameters_weight_,
            l_self_modules_features_modules_16_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_152 = (
            l_self_modules_features_modules_16_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_16_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_features_modules_16_modules_bn_parameters_weight_
        ) = l_self_modules_features_modules_16_modules_bn_parameters_bias_ = None
        x_154 = torch.nn.functional.silu(x_153, inplace=True)
        x_153 = None
        x_155 = torch.nn.functional.adaptive_avg_pool2d(x_154, 1)
        x_154 = None
        x_156 = x_155.flatten(1, -1)
        x_155 = None
        x_157 = torch.nn.functional.dropout(x_156, 0.2, False, False)
        x_156 = None
        x_158 = torch._C._nn.linear(
            x_157,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_157 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_158,)
