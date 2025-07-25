import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_features_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_features_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_conv_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_conv_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_conv_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_conv_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_conv_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_conv_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_conv_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_conv_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_conv_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_conv_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_conv_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_conv_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_conv_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_conv_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_conv_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_conv_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_conv_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_conv_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_conv_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_conv_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_conv_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_conv_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_conv_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_conv_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_conv_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_conv_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_conv_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_conv_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_conv_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_conv_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_modules_conv_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_modules_conv_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_conv_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_conv_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_conv_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_modules_conv_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_modules_conv_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_conv_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_conv_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_conv_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_modules_conv_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_modules_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_conv_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_conv_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_9_modules_conv_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_9_modules_conv_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_conv_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_conv_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_conv_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_9_modules_conv_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_9_modules_conv_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_conv_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_conv_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_conv_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_9_modules_conv_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_9_modules_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_conv_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_conv_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_10_modules_conv_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_10_modules_conv_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_conv_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_conv_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_conv_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_10_modules_conv_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_10_modules_conv_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_conv_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_conv_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_conv_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_10_modules_conv_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_10_modules_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_conv_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_conv_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_conv_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_conv_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_conv_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_conv_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_conv_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_conv_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_conv_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_conv_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_conv_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_conv_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_conv_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_conv_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_conv_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_modules_conv_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_modules_conv_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_conv_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_conv_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_conv_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_modules_conv_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_modules_conv_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_conv_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_conv_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_conv_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_modules_conv_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_modules_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_conv_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_conv_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_13_modules_conv_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_13_modules_conv_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_conv_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_conv_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_conv_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_13_modules_conv_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_13_modules_conv_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_conv_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_conv_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_conv_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_13_modules_conv_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_13_modules_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_13_modules_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_conv_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_conv_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_14_modules_conv_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_14_modules_conv_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_conv_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_conv_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_conv_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_14_modules_conv_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_14_modules_conv_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_conv_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_conv_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_conv_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_14_modules_conv_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_14_modules_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_14_modules_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_conv_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_conv_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_15_modules_conv_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_15_modules_conv_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_conv_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_conv_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_conv_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_15_modules_conv_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_15_modules_conv_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_conv_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_conv_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_conv_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_15_modules_conv_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_15_modules_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_15_modules_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_conv_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_conv_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_16_modules_conv_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_16_modules_conv_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_conv_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_conv_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_conv_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_16_modules_conv_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_16_modules_conv_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_conv_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_conv_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_conv_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_16_modules_conv_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_16_modules_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_16_modules_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_conv_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_conv_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_17_modules_conv_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_17_modules_conv_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_conv_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_conv_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_conv_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_17_modules_conv_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_17_modules_conv_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_conv_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_conv_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_conv_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_17_modules_conv_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_17_modules_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_17_modules_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_18_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_18_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_18_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_features_modules_0_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_0_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_features_modules_0_modules_1_buffers_running_mean_ = (
            L_self_modules_features_modules_0_modules_1_buffers_running_mean_
        )
        l_self_modules_features_modules_0_modules_1_buffers_running_var_ = (
            L_self_modules_features_modules_0_modules_1_buffers_running_var_
        )
        l_self_modules_features_modules_0_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_0_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_0_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_0_modules_1_parameters_bias_
        )
        l_self_modules_features_modules_1_modules_conv_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_conv_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_conv_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_conv_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_conv_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_conv_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_conv_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_1_modules_conv_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_1_modules_conv_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_1_modules_conv_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_1_modules_conv_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_conv_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_conv_modules_2_buffers_running_mean_ = L_self_modules_features_modules_1_modules_conv_modules_2_buffers_running_mean_
        l_self_modules_features_modules_1_modules_conv_modules_2_buffers_running_var_ = L_self_modules_features_modules_1_modules_conv_modules_2_buffers_running_var_
        l_self_modules_features_modules_1_modules_conv_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_1_modules_conv_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_1_modules_conv_modules_2_parameters_bias_ = (
            L_self_modules_features_modules_1_modules_conv_modules_2_parameters_bias_
        )
        l_self_modules_features_modules_2_modules_conv_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_conv_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_conv_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_conv_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_conv_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_conv_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_conv_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_conv_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_conv_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_conv_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_conv_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_conv_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_conv_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_conv_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_conv_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_conv_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_conv_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_conv_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_conv_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_conv_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_conv_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_conv_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_conv_modules_3_buffers_running_mean_ = L_self_modules_features_modules_2_modules_conv_modules_3_buffers_running_mean_
        l_self_modules_features_modules_2_modules_conv_modules_3_buffers_running_var_ = L_self_modules_features_modules_2_modules_conv_modules_3_buffers_running_var_
        l_self_modules_features_modules_2_modules_conv_modules_3_parameters_weight_ = (
            L_self_modules_features_modules_2_modules_conv_modules_3_parameters_weight_
        )
        l_self_modules_features_modules_2_modules_conv_modules_3_parameters_bias_ = (
            L_self_modules_features_modules_2_modules_conv_modules_3_parameters_bias_
        )
        l_self_modules_features_modules_3_modules_conv_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_conv_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_conv_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_conv_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_conv_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_conv_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_conv_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_conv_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_conv_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_conv_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_conv_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_conv_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_conv_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_conv_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_conv_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_conv_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_conv_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_conv_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_conv_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_conv_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_conv_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_conv_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_conv_modules_3_buffers_running_mean_ = L_self_modules_features_modules_3_modules_conv_modules_3_buffers_running_mean_
        l_self_modules_features_modules_3_modules_conv_modules_3_buffers_running_var_ = L_self_modules_features_modules_3_modules_conv_modules_3_buffers_running_var_
        l_self_modules_features_modules_3_modules_conv_modules_3_parameters_weight_ = (
            L_self_modules_features_modules_3_modules_conv_modules_3_parameters_weight_
        )
        l_self_modules_features_modules_3_modules_conv_modules_3_parameters_bias_ = (
            L_self_modules_features_modules_3_modules_conv_modules_3_parameters_bias_
        )
        l_self_modules_features_modules_4_modules_conv_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_conv_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_conv_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_conv_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_conv_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_conv_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_conv_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_conv_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_conv_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_conv_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_conv_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_conv_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_conv_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_conv_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_conv_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_conv_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_conv_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_conv_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_conv_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_conv_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_conv_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_conv_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_conv_modules_3_buffers_running_mean_ = L_self_modules_features_modules_4_modules_conv_modules_3_buffers_running_mean_
        l_self_modules_features_modules_4_modules_conv_modules_3_buffers_running_var_ = L_self_modules_features_modules_4_modules_conv_modules_3_buffers_running_var_
        l_self_modules_features_modules_4_modules_conv_modules_3_parameters_weight_ = (
            L_self_modules_features_modules_4_modules_conv_modules_3_parameters_weight_
        )
        l_self_modules_features_modules_4_modules_conv_modules_3_parameters_bias_ = (
            L_self_modules_features_modules_4_modules_conv_modules_3_parameters_bias_
        )
        l_self_modules_features_modules_5_modules_conv_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_conv_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_conv_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_conv_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_conv_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_conv_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_conv_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_conv_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_conv_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_conv_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_conv_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_conv_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_conv_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_conv_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_conv_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_conv_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_conv_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_conv_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_conv_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_conv_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_conv_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_conv_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_conv_modules_3_buffers_running_mean_ = L_self_modules_features_modules_5_modules_conv_modules_3_buffers_running_mean_
        l_self_modules_features_modules_5_modules_conv_modules_3_buffers_running_var_ = L_self_modules_features_modules_5_modules_conv_modules_3_buffers_running_var_
        l_self_modules_features_modules_5_modules_conv_modules_3_parameters_weight_ = (
            L_self_modules_features_modules_5_modules_conv_modules_3_parameters_weight_
        )
        l_self_modules_features_modules_5_modules_conv_modules_3_parameters_bias_ = (
            L_self_modules_features_modules_5_modules_conv_modules_3_parameters_bias_
        )
        l_self_modules_features_modules_6_modules_conv_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_conv_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_conv_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_conv_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_conv_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_conv_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_conv_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_conv_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_conv_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_conv_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_conv_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_conv_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_conv_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_conv_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_conv_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_conv_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_conv_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_conv_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_conv_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_conv_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_conv_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_6_modules_conv_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_6_modules_conv_modules_3_buffers_running_mean_ = L_self_modules_features_modules_6_modules_conv_modules_3_buffers_running_mean_
        l_self_modules_features_modules_6_modules_conv_modules_3_buffers_running_var_ = L_self_modules_features_modules_6_modules_conv_modules_3_buffers_running_var_
        l_self_modules_features_modules_6_modules_conv_modules_3_parameters_weight_ = (
            L_self_modules_features_modules_6_modules_conv_modules_3_parameters_weight_
        )
        l_self_modules_features_modules_6_modules_conv_modules_3_parameters_bias_ = (
            L_self_modules_features_modules_6_modules_conv_modules_3_parameters_bias_
        )
        l_self_modules_features_modules_7_modules_conv_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_conv_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_conv_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_conv_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_conv_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_conv_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_conv_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_conv_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_conv_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_conv_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_conv_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_conv_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_conv_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_conv_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_conv_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_conv_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_conv_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_conv_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_conv_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_conv_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_conv_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_7_modules_conv_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_conv_modules_3_buffers_running_mean_ = L_self_modules_features_modules_7_modules_conv_modules_3_buffers_running_mean_
        l_self_modules_features_modules_7_modules_conv_modules_3_buffers_running_var_ = L_self_modules_features_modules_7_modules_conv_modules_3_buffers_running_var_
        l_self_modules_features_modules_7_modules_conv_modules_3_parameters_weight_ = (
            L_self_modules_features_modules_7_modules_conv_modules_3_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_conv_modules_3_parameters_bias_ = (
            L_self_modules_features_modules_7_modules_conv_modules_3_parameters_bias_
        )
        l_self_modules_features_modules_8_modules_conv_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_8_modules_conv_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_8_modules_conv_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_8_modules_conv_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_8_modules_conv_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_8_modules_conv_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_8_modules_conv_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_8_modules_conv_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_8_modules_conv_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_8_modules_conv_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_8_modules_conv_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_8_modules_conv_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_8_modules_conv_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_8_modules_conv_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_8_modules_conv_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_8_modules_conv_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_8_modules_conv_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_8_modules_conv_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_8_modules_conv_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_8_modules_conv_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_8_modules_conv_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_8_modules_conv_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_8_modules_conv_modules_3_buffers_running_mean_ = L_self_modules_features_modules_8_modules_conv_modules_3_buffers_running_mean_
        l_self_modules_features_modules_8_modules_conv_modules_3_buffers_running_var_ = L_self_modules_features_modules_8_modules_conv_modules_3_buffers_running_var_
        l_self_modules_features_modules_8_modules_conv_modules_3_parameters_weight_ = (
            L_self_modules_features_modules_8_modules_conv_modules_3_parameters_weight_
        )
        l_self_modules_features_modules_8_modules_conv_modules_3_parameters_bias_ = (
            L_self_modules_features_modules_8_modules_conv_modules_3_parameters_bias_
        )
        l_self_modules_features_modules_9_modules_conv_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_9_modules_conv_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_9_modules_conv_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_9_modules_conv_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_9_modules_conv_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_9_modules_conv_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_9_modules_conv_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_9_modules_conv_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_9_modules_conv_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_9_modules_conv_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_9_modules_conv_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_9_modules_conv_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_9_modules_conv_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_9_modules_conv_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_9_modules_conv_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_9_modules_conv_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_9_modules_conv_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_9_modules_conv_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_9_modules_conv_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_9_modules_conv_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_9_modules_conv_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_9_modules_conv_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_9_modules_conv_modules_3_buffers_running_mean_ = L_self_modules_features_modules_9_modules_conv_modules_3_buffers_running_mean_
        l_self_modules_features_modules_9_modules_conv_modules_3_buffers_running_var_ = L_self_modules_features_modules_9_modules_conv_modules_3_buffers_running_var_
        l_self_modules_features_modules_9_modules_conv_modules_3_parameters_weight_ = (
            L_self_modules_features_modules_9_modules_conv_modules_3_parameters_weight_
        )
        l_self_modules_features_modules_9_modules_conv_modules_3_parameters_bias_ = (
            L_self_modules_features_modules_9_modules_conv_modules_3_parameters_bias_
        )
        l_self_modules_features_modules_10_modules_conv_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_10_modules_conv_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_10_modules_conv_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_10_modules_conv_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_10_modules_conv_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_10_modules_conv_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_10_modules_conv_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_10_modules_conv_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_10_modules_conv_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_10_modules_conv_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_10_modules_conv_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_10_modules_conv_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_10_modules_conv_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_10_modules_conv_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_10_modules_conv_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_10_modules_conv_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_10_modules_conv_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_10_modules_conv_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_10_modules_conv_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_10_modules_conv_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_10_modules_conv_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_10_modules_conv_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_10_modules_conv_modules_3_buffers_running_mean_ = L_self_modules_features_modules_10_modules_conv_modules_3_buffers_running_mean_
        l_self_modules_features_modules_10_modules_conv_modules_3_buffers_running_var_ = L_self_modules_features_modules_10_modules_conv_modules_3_buffers_running_var_
        l_self_modules_features_modules_10_modules_conv_modules_3_parameters_weight_ = (
            L_self_modules_features_modules_10_modules_conv_modules_3_parameters_weight_
        )
        l_self_modules_features_modules_10_modules_conv_modules_3_parameters_bias_ = (
            L_self_modules_features_modules_10_modules_conv_modules_3_parameters_bias_
        )
        l_self_modules_features_modules_11_modules_conv_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_11_modules_conv_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_11_modules_conv_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_11_modules_conv_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_11_modules_conv_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_11_modules_conv_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_11_modules_conv_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_11_modules_conv_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_11_modules_conv_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_11_modules_conv_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_11_modules_conv_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_11_modules_conv_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_11_modules_conv_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_11_modules_conv_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_11_modules_conv_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_11_modules_conv_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_11_modules_conv_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_11_modules_conv_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_11_modules_conv_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_11_modules_conv_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_11_modules_conv_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_11_modules_conv_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_11_modules_conv_modules_3_buffers_running_mean_ = L_self_modules_features_modules_11_modules_conv_modules_3_buffers_running_mean_
        l_self_modules_features_modules_11_modules_conv_modules_3_buffers_running_var_ = L_self_modules_features_modules_11_modules_conv_modules_3_buffers_running_var_
        l_self_modules_features_modules_11_modules_conv_modules_3_parameters_weight_ = (
            L_self_modules_features_modules_11_modules_conv_modules_3_parameters_weight_
        )
        l_self_modules_features_modules_11_modules_conv_modules_3_parameters_bias_ = (
            L_self_modules_features_modules_11_modules_conv_modules_3_parameters_bias_
        )
        l_self_modules_features_modules_12_modules_conv_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_12_modules_conv_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_12_modules_conv_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_12_modules_conv_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_12_modules_conv_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_12_modules_conv_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_12_modules_conv_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_12_modules_conv_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_12_modules_conv_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_12_modules_conv_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_12_modules_conv_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_12_modules_conv_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_12_modules_conv_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_12_modules_conv_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_12_modules_conv_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_12_modules_conv_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_12_modules_conv_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_12_modules_conv_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_12_modules_conv_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_12_modules_conv_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_12_modules_conv_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_12_modules_conv_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_12_modules_conv_modules_3_buffers_running_mean_ = L_self_modules_features_modules_12_modules_conv_modules_3_buffers_running_mean_
        l_self_modules_features_modules_12_modules_conv_modules_3_buffers_running_var_ = L_self_modules_features_modules_12_modules_conv_modules_3_buffers_running_var_
        l_self_modules_features_modules_12_modules_conv_modules_3_parameters_weight_ = (
            L_self_modules_features_modules_12_modules_conv_modules_3_parameters_weight_
        )
        l_self_modules_features_modules_12_modules_conv_modules_3_parameters_bias_ = (
            L_self_modules_features_modules_12_modules_conv_modules_3_parameters_bias_
        )
        l_self_modules_features_modules_13_modules_conv_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_13_modules_conv_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_13_modules_conv_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_13_modules_conv_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_13_modules_conv_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_13_modules_conv_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_13_modules_conv_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_13_modules_conv_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_13_modules_conv_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_13_modules_conv_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_13_modules_conv_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_13_modules_conv_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_13_modules_conv_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_13_modules_conv_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_13_modules_conv_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_13_modules_conv_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_13_modules_conv_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_13_modules_conv_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_13_modules_conv_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_13_modules_conv_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_13_modules_conv_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_13_modules_conv_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_13_modules_conv_modules_3_buffers_running_mean_ = L_self_modules_features_modules_13_modules_conv_modules_3_buffers_running_mean_
        l_self_modules_features_modules_13_modules_conv_modules_3_buffers_running_var_ = L_self_modules_features_modules_13_modules_conv_modules_3_buffers_running_var_
        l_self_modules_features_modules_13_modules_conv_modules_3_parameters_weight_ = (
            L_self_modules_features_modules_13_modules_conv_modules_3_parameters_weight_
        )
        l_self_modules_features_modules_13_modules_conv_modules_3_parameters_bias_ = (
            L_self_modules_features_modules_13_modules_conv_modules_3_parameters_bias_
        )
        l_self_modules_features_modules_14_modules_conv_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_14_modules_conv_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_14_modules_conv_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_14_modules_conv_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_14_modules_conv_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_14_modules_conv_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_14_modules_conv_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_14_modules_conv_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_14_modules_conv_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_14_modules_conv_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_14_modules_conv_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_14_modules_conv_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_14_modules_conv_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_14_modules_conv_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_14_modules_conv_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_14_modules_conv_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_14_modules_conv_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_14_modules_conv_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_14_modules_conv_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_14_modules_conv_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_14_modules_conv_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_14_modules_conv_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_14_modules_conv_modules_3_buffers_running_mean_ = L_self_modules_features_modules_14_modules_conv_modules_3_buffers_running_mean_
        l_self_modules_features_modules_14_modules_conv_modules_3_buffers_running_var_ = L_self_modules_features_modules_14_modules_conv_modules_3_buffers_running_var_
        l_self_modules_features_modules_14_modules_conv_modules_3_parameters_weight_ = (
            L_self_modules_features_modules_14_modules_conv_modules_3_parameters_weight_
        )
        l_self_modules_features_modules_14_modules_conv_modules_3_parameters_bias_ = (
            L_self_modules_features_modules_14_modules_conv_modules_3_parameters_bias_
        )
        l_self_modules_features_modules_15_modules_conv_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_15_modules_conv_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_15_modules_conv_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_15_modules_conv_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_15_modules_conv_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_15_modules_conv_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_15_modules_conv_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_15_modules_conv_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_15_modules_conv_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_15_modules_conv_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_15_modules_conv_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_15_modules_conv_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_15_modules_conv_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_15_modules_conv_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_15_modules_conv_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_15_modules_conv_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_15_modules_conv_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_15_modules_conv_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_15_modules_conv_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_15_modules_conv_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_15_modules_conv_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_15_modules_conv_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_15_modules_conv_modules_3_buffers_running_mean_ = L_self_modules_features_modules_15_modules_conv_modules_3_buffers_running_mean_
        l_self_modules_features_modules_15_modules_conv_modules_3_buffers_running_var_ = L_self_modules_features_modules_15_modules_conv_modules_3_buffers_running_var_
        l_self_modules_features_modules_15_modules_conv_modules_3_parameters_weight_ = (
            L_self_modules_features_modules_15_modules_conv_modules_3_parameters_weight_
        )
        l_self_modules_features_modules_15_modules_conv_modules_3_parameters_bias_ = (
            L_self_modules_features_modules_15_modules_conv_modules_3_parameters_bias_
        )
        l_self_modules_features_modules_16_modules_conv_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_16_modules_conv_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_16_modules_conv_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_16_modules_conv_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_16_modules_conv_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_16_modules_conv_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_16_modules_conv_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_16_modules_conv_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_16_modules_conv_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_16_modules_conv_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_16_modules_conv_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_16_modules_conv_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_16_modules_conv_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_16_modules_conv_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_16_modules_conv_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_16_modules_conv_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_16_modules_conv_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_16_modules_conv_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_16_modules_conv_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_16_modules_conv_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_16_modules_conv_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_16_modules_conv_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_16_modules_conv_modules_3_buffers_running_mean_ = L_self_modules_features_modules_16_modules_conv_modules_3_buffers_running_mean_
        l_self_modules_features_modules_16_modules_conv_modules_3_buffers_running_var_ = L_self_modules_features_modules_16_modules_conv_modules_3_buffers_running_var_
        l_self_modules_features_modules_16_modules_conv_modules_3_parameters_weight_ = (
            L_self_modules_features_modules_16_modules_conv_modules_3_parameters_weight_
        )
        l_self_modules_features_modules_16_modules_conv_modules_3_parameters_bias_ = (
            L_self_modules_features_modules_16_modules_conv_modules_3_parameters_bias_
        )
        l_self_modules_features_modules_17_modules_conv_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_17_modules_conv_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_17_modules_conv_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_17_modules_conv_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_17_modules_conv_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_17_modules_conv_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_17_modules_conv_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_17_modules_conv_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_17_modules_conv_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_17_modules_conv_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_17_modules_conv_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_17_modules_conv_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_17_modules_conv_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_17_modules_conv_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_17_modules_conv_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_17_modules_conv_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_17_modules_conv_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_17_modules_conv_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_17_modules_conv_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_17_modules_conv_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_17_modules_conv_modules_2_parameters_weight_ = (
            L_self_modules_features_modules_17_modules_conv_modules_2_parameters_weight_
        )
        l_self_modules_features_modules_17_modules_conv_modules_3_buffers_running_mean_ = L_self_modules_features_modules_17_modules_conv_modules_3_buffers_running_mean_
        l_self_modules_features_modules_17_modules_conv_modules_3_buffers_running_var_ = L_self_modules_features_modules_17_modules_conv_modules_3_buffers_running_var_
        l_self_modules_features_modules_17_modules_conv_modules_3_parameters_weight_ = (
            L_self_modules_features_modules_17_modules_conv_modules_3_parameters_weight_
        )
        l_self_modules_features_modules_17_modules_conv_modules_3_parameters_bias_ = (
            L_self_modules_features_modules_17_modules_conv_modules_3_parameters_bias_
        )
        l_self_modules_features_modules_18_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_18_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_18_modules_1_buffers_running_mean_ = (
            L_self_modules_features_modules_18_modules_1_buffers_running_mean_
        )
        l_self_modules_features_modules_18_modules_1_buffers_running_var_ = (
            L_self_modules_features_modules_18_modules_1_buffers_running_var_
        )
        l_self_modules_features_modules_18_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_18_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_18_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_18_modules_1_parameters_bias_
        )
        l_self_modules_classifier_modules_1_parameters_weight_ = (
            L_self_modules_classifier_modules_1_parameters_weight_
        )
        l_self_modules_classifier_modules_1_parameters_bias_ = (
            L_self_modules_classifier_modules_1_parameters_bias_
        )
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_features_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_features_modules_0_modules_0_parameters_weight_ = None
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_features_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = (
            l_self_modules_features_modules_0_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_0_modules_1_buffers_running_var_
        ) = (
            l_self_modules_features_modules_0_modules_1_parameters_weight_
        ) = l_self_modules_features_modules_0_modules_1_parameters_bias_ = None
        input_3 = torch.nn.functional.hardtanh(input_2, 0.0, 6.0, True)
        input_2 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_features_modules_1_modules_conv_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        input_3 = l_self_modules_features_modules_1_modules_conv_modules_0_modules_0_parameters_weight_ = (None)
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_features_modules_1_modules_conv_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_conv_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_conv_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_conv_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_4 = l_self_modules_features_modules_1_modules_conv_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_conv_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_1_modules_conv_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_1_modules_conv_modules_0_modules_1_parameters_bias_ = (None)
        input_6 = torch.nn.functional.hardtanh(input_5, 0.0, 6.0, True)
        input_5 = None
        input_7 = torch.conv2d(
            input_6,
            l_self_modules_features_modules_1_modules_conv_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_6 = (
            l_self_modules_features_modules_1_modules_conv_modules_1_parameters_weight_
        ) = None
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_features_modules_1_modules_conv_modules_2_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_conv_modules_2_buffers_running_var_,
            l_self_modules_features_modules_1_modules_conv_modules_2_parameters_weight_,
            l_self_modules_features_modules_1_modules_conv_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = l_self_modules_features_modules_1_modules_conv_modules_2_buffers_running_mean_ = l_self_modules_features_modules_1_modules_conv_modules_2_buffers_running_var_ = (
            l_self_modules_features_modules_1_modules_conv_modules_2_parameters_weight_
        ) = (
            l_self_modules_features_modules_1_modules_conv_modules_2_parameters_bias_
        ) = None
        input_9 = torch.conv2d(
            input_8,
            l_self_modules_features_modules_2_modules_conv_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_8 = l_self_modules_features_modules_2_modules_conv_modules_0_modules_0_parameters_weight_ = (None)
        input_10 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_features_modules_2_modules_conv_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_conv_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_conv_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_conv_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_9 = l_self_modules_features_modules_2_modules_conv_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_conv_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_conv_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_conv_modules_0_modules_1_parameters_bias_ = (None)
        input_11 = torch.nn.functional.hardtanh(input_10, 0.0, 6.0, True)
        input_10 = None
        input_12 = torch.conv2d(
            input_11,
            l_self_modules_features_modules_2_modules_conv_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            96,
        )
        input_11 = l_self_modules_features_modules_2_modules_conv_modules_1_modules_0_parameters_weight_ = (None)
        input_13 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_features_modules_2_modules_conv_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_conv_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_conv_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_conv_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = l_self_modules_features_modules_2_modules_conv_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_conv_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_conv_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_conv_modules_1_modules_1_parameters_bias_ = (None)
        input_14 = torch.nn.functional.hardtanh(input_13, 0.0, 6.0, True)
        input_13 = None
        input_15 = torch.conv2d(
            input_14,
            l_self_modules_features_modules_2_modules_conv_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_14 = (
            l_self_modules_features_modules_2_modules_conv_modules_2_parameters_weight_
        ) = None
        input_16 = torch.nn.functional.batch_norm(
            input_15,
            l_self_modules_features_modules_2_modules_conv_modules_3_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_conv_modules_3_buffers_running_var_,
            l_self_modules_features_modules_2_modules_conv_modules_3_parameters_weight_,
            l_self_modules_features_modules_2_modules_conv_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_15 = l_self_modules_features_modules_2_modules_conv_modules_3_buffers_running_mean_ = l_self_modules_features_modules_2_modules_conv_modules_3_buffers_running_var_ = (
            l_self_modules_features_modules_2_modules_conv_modules_3_parameters_weight_
        ) = (
            l_self_modules_features_modules_2_modules_conv_modules_3_parameters_bias_
        ) = None
        input_17 = torch.conv2d(
            input_16,
            l_self_modules_features_modules_3_modules_conv_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_3_modules_conv_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_18 = torch.nn.functional.batch_norm(
            input_17,
            l_self_modules_features_modules_3_modules_conv_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_conv_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_conv_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_conv_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_17 = l_self_modules_features_modules_3_modules_conv_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_conv_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_conv_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_conv_modules_0_modules_1_parameters_bias_ = (None)
        input_19 = torch.nn.functional.hardtanh(input_18, 0.0, 6.0, True)
        input_18 = None
        input_20 = torch.conv2d(
            input_19,
            l_self_modules_features_modules_3_modules_conv_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            144,
        )
        input_19 = l_self_modules_features_modules_3_modules_conv_modules_1_modules_0_parameters_weight_ = (None)
        input_21 = torch.nn.functional.batch_norm(
            input_20,
            l_self_modules_features_modules_3_modules_conv_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_conv_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_conv_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_conv_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_20 = l_self_modules_features_modules_3_modules_conv_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_conv_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_conv_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_conv_modules_1_modules_1_parameters_bias_ = (None)
        input_22 = torch.nn.functional.hardtanh(input_21, 0.0, 6.0, True)
        input_21 = None
        input_23 = torch.conv2d(
            input_22,
            l_self_modules_features_modules_3_modules_conv_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_22 = (
            l_self_modules_features_modules_3_modules_conv_modules_2_parameters_weight_
        ) = None
        input_24 = torch.nn.functional.batch_norm(
            input_23,
            l_self_modules_features_modules_3_modules_conv_modules_3_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_conv_modules_3_buffers_running_var_,
            l_self_modules_features_modules_3_modules_conv_modules_3_parameters_weight_,
            l_self_modules_features_modules_3_modules_conv_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_23 = l_self_modules_features_modules_3_modules_conv_modules_3_buffers_running_mean_ = l_self_modules_features_modules_3_modules_conv_modules_3_buffers_running_var_ = (
            l_self_modules_features_modules_3_modules_conv_modules_3_parameters_weight_
        ) = (
            l_self_modules_features_modules_3_modules_conv_modules_3_parameters_bias_
        ) = None
        input_25 = input_16 + input_24
        input_16 = input_24 = None
        input_26 = torch.conv2d(
            input_25,
            l_self_modules_features_modules_4_modules_conv_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_25 = l_self_modules_features_modules_4_modules_conv_modules_0_modules_0_parameters_weight_ = (None)
        input_27 = torch.nn.functional.batch_norm(
            input_26,
            l_self_modules_features_modules_4_modules_conv_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_conv_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_conv_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_conv_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_26 = l_self_modules_features_modules_4_modules_conv_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_conv_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_conv_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_conv_modules_0_modules_1_parameters_bias_ = (None)
        input_28 = torch.nn.functional.hardtanh(input_27, 0.0, 6.0, True)
        input_27 = None
        input_29 = torch.conv2d(
            input_28,
            l_self_modules_features_modules_4_modules_conv_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            144,
        )
        input_28 = l_self_modules_features_modules_4_modules_conv_modules_1_modules_0_parameters_weight_ = (None)
        input_30 = torch.nn.functional.batch_norm(
            input_29,
            l_self_modules_features_modules_4_modules_conv_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_conv_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_conv_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_conv_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_29 = l_self_modules_features_modules_4_modules_conv_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_conv_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_conv_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_conv_modules_1_modules_1_parameters_bias_ = (None)
        input_31 = torch.nn.functional.hardtanh(input_30, 0.0, 6.0, True)
        input_30 = None
        input_32 = torch.conv2d(
            input_31,
            l_self_modules_features_modules_4_modules_conv_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_31 = (
            l_self_modules_features_modules_4_modules_conv_modules_2_parameters_weight_
        ) = None
        input_33 = torch.nn.functional.batch_norm(
            input_32,
            l_self_modules_features_modules_4_modules_conv_modules_3_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_conv_modules_3_buffers_running_var_,
            l_self_modules_features_modules_4_modules_conv_modules_3_parameters_weight_,
            l_self_modules_features_modules_4_modules_conv_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_32 = l_self_modules_features_modules_4_modules_conv_modules_3_buffers_running_mean_ = l_self_modules_features_modules_4_modules_conv_modules_3_buffers_running_var_ = (
            l_self_modules_features_modules_4_modules_conv_modules_3_parameters_weight_
        ) = (
            l_self_modules_features_modules_4_modules_conv_modules_3_parameters_bias_
        ) = None
        input_34 = torch.conv2d(
            input_33,
            l_self_modules_features_modules_5_modules_conv_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_conv_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_35 = torch.nn.functional.batch_norm(
            input_34,
            l_self_modules_features_modules_5_modules_conv_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_conv_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_conv_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_conv_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_34 = l_self_modules_features_modules_5_modules_conv_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_conv_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_conv_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_conv_modules_0_modules_1_parameters_bias_ = (None)
        input_36 = torch.nn.functional.hardtanh(input_35, 0.0, 6.0, True)
        input_35 = None
        input_37 = torch.conv2d(
            input_36,
            l_self_modules_features_modules_5_modules_conv_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        input_36 = l_self_modules_features_modules_5_modules_conv_modules_1_modules_0_parameters_weight_ = (None)
        input_38 = torch.nn.functional.batch_norm(
            input_37,
            l_self_modules_features_modules_5_modules_conv_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_conv_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_conv_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_conv_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_37 = l_self_modules_features_modules_5_modules_conv_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_conv_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_conv_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_conv_modules_1_modules_1_parameters_bias_ = (None)
        input_39 = torch.nn.functional.hardtanh(input_38, 0.0, 6.0, True)
        input_38 = None
        input_40 = torch.conv2d(
            input_39,
            l_self_modules_features_modules_5_modules_conv_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_39 = (
            l_self_modules_features_modules_5_modules_conv_modules_2_parameters_weight_
        ) = None
        input_41 = torch.nn.functional.batch_norm(
            input_40,
            l_self_modules_features_modules_5_modules_conv_modules_3_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_conv_modules_3_buffers_running_var_,
            l_self_modules_features_modules_5_modules_conv_modules_3_parameters_weight_,
            l_self_modules_features_modules_5_modules_conv_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_40 = l_self_modules_features_modules_5_modules_conv_modules_3_buffers_running_mean_ = l_self_modules_features_modules_5_modules_conv_modules_3_buffers_running_var_ = (
            l_self_modules_features_modules_5_modules_conv_modules_3_parameters_weight_
        ) = (
            l_self_modules_features_modules_5_modules_conv_modules_3_parameters_bias_
        ) = None
        input_42 = input_33 + input_41
        input_33 = input_41 = None
        input_43 = torch.conv2d(
            input_42,
            l_self_modules_features_modules_6_modules_conv_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_conv_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_44 = torch.nn.functional.batch_norm(
            input_43,
            l_self_modules_features_modules_6_modules_conv_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_conv_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_conv_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_conv_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_43 = l_self_modules_features_modules_6_modules_conv_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_conv_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_conv_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_conv_modules_0_modules_1_parameters_bias_ = (None)
        input_45 = torch.nn.functional.hardtanh(input_44, 0.0, 6.0, True)
        input_44 = None
        input_46 = torch.conv2d(
            input_45,
            l_self_modules_features_modules_6_modules_conv_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            192,
        )
        input_45 = l_self_modules_features_modules_6_modules_conv_modules_1_modules_0_parameters_weight_ = (None)
        input_47 = torch.nn.functional.batch_norm(
            input_46,
            l_self_modules_features_modules_6_modules_conv_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_conv_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_conv_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_conv_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_46 = l_self_modules_features_modules_6_modules_conv_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_conv_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_conv_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_conv_modules_1_modules_1_parameters_bias_ = (None)
        input_48 = torch.nn.functional.hardtanh(input_47, 0.0, 6.0, True)
        input_47 = None
        input_49 = torch.conv2d(
            input_48,
            l_self_modules_features_modules_6_modules_conv_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_48 = (
            l_self_modules_features_modules_6_modules_conv_modules_2_parameters_weight_
        ) = None
        input_50 = torch.nn.functional.batch_norm(
            input_49,
            l_self_modules_features_modules_6_modules_conv_modules_3_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_conv_modules_3_buffers_running_var_,
            l_self_modules_features_modules_6_modules_conv_modules_3_parameters_weight_,
            l_self_modules_features_modules_6_modules_conv_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_49 = l_self_modules_features_modules_6_modules_conv_modules_3_buffers_running_mean_ = l_self_modules_features_modules_6_modules_conv_modules_3_buffers_running_var_ = (
            l_self_modules_features_modules_6_modules_conv_modules_3_parameters_weight_
        ) = (
            l_self_modules_features_modules_6_modules_conv_modules_3_parameters_bias_
        ) = None
        input_51 = input_42 + input_50
        input_42 = input_50 = None
        input_52 = torch.conv2d(
            input_51,
            l_self_modules_features_modules_7_modules_conv_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_51 = l_self_modules_features_modules_7_modules_conv_modules_0_modules_0_parameters_weight_ = (None)
        input_53 = torch.nn.functional.batch_norm(
            input_52,
            l_self_modules_features_modules_7_modules_conv_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_conv_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_conv_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_conv_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_52 = l_self_modules_features_modules_7_modules_conv_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_conv_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_conv_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_conv_modules_0_modules_1_parameters_bias_ = (None)
        input_54 = torch.nn.functional.hardtanh(input_53, 0.0, 6.0, True)
        input_53 = None
        input_55 = torch.conv2d(
            input_54,
            l_self_modules_features_modules_7_modules_conv_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            192,
        )
        input_54 = l_self_modules_features_modules_7_modules_conv_modules_1_modules_0_parameters_weight_ = (None)
        input_56 = torch.nn.functional.batch_norm(
            input_55,
            l_self_modules_features_modules_7_modules_conv_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_conv_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_conv_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_conv_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_55 = l_self_modules_features_modules_7_modules_conv_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_conv_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_conv_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_conv_modules_1_modules_1_parameters_bias_ = (None)
        input_57 = torch.nn.functional.hardtanh(input_56, 0.0, 6.0, True)
        input_56 = None
        input_58 = torch.conv2d(
            input_57,
            l_self_modules_features_modules_7_modules_conv_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_57 = (
            l_self_modules_features_modules_7_modules_conv_modules_2_parameters_weight_
        ) = None
        input_59 = torch.nn.functional.batch_norm(
            input_58,
            l_self_modules_features_modules_7_modules_conv_modules_3_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_conv_modules_3_buffers_running_var_,
            l_self_modules_features_modules_7_modules_conv_modules_3_parameters_weight_,
            l_self_modules_features_modules_7_modules_conv_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_58 = l_self_modules_features_modules_7_modules_conv_modules_3_buffers_running_mean_ = l_self_modules_features_modules_7_modules_conv_modules_3_buffers_running_var_ = (
            l_self_modules_features_modules_7_modules_conv_modules_3_parameters_weight_
        ) = (
            l_self_modules_features_modules_7_modules_conv_modules_3_parameters_bias_
        ) = None
        input_60 = torch.conv2d(
            input_59,
            l_self_modules_features_modules_8_modules_conv_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_8_modules_conv_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_61 = torch.nn.functional.batch_norm(
            input_60,
            l_self_modules_features_modules_8_modules_conv_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_conv_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_8_modules_conv_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_8_modules_conv_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_60 = l_self_modules_features_modules_8_modules_conv_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_8_modules_conv_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_8_modules_conv_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_8_modules_conv_modules_0_modules_1_parameters_bias_ = (None)
        input_62 = torch.nn.functional.hardtanh(input_61, 0.0, 6.0, True)
        input_61 = None
        input_63 = torch.conv2d(
            input_62,
            l_self_modules_features_modules_8_modules_conv_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        input_62 = l_self_modules_features_modules_8_modules_conv_modules_1_modules_0_parameters_weight_ = (None)
        input_64 = torch.nn.functional.batch_norm(
            input_63,
            l_self_modules_features_modules_8_modules_conv_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_conv_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_8_modules_conv_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_8_modules_conv_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_63 = l_self_modules_features_modules_8_modules_conv_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_8_modules_conv_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_8_modules_conv_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_8_modules_conv_modules_1_modules_1_parameters_bias_ = (None)
        input_65 = torch.nn.functional.hardtanh(input_64, 0.0, 6.0, True)
        input_64 = None
        input_66 = torch.conv2d(
            input_65,
            l_self_modules_features_modules_8_modules_conv_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_65 = (
            l_self_modules_features_modules_8_modules_conv_modules_2_parameters_weight_
        ) = None
        input_67 = torch.nn.functional.batch_norm(
            input_66,
            l_self_modules_features_modules_8_modules_conv_modules_3_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_conv_modules_3_buffers_running_var_,
            l_self_modules_features_modules_8_modules_conv_modules_3_parameters_weight_,
            l_self_modules_features_modules_8_modules_conv_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_66 = l_self_modules_features_modules_8_modules_conv_modules_3_buffers_running_mean_ = l_self_modules_features_modules_8_modules_conv_modules_3_buffers_running_var_ = (
            l_self_modules_features_modules_8_modules_conv_modules_3_parameters_weight_
        ) = (
            l_self_modules_features_modules_8_modules_conv_modules_3_parameters_bias_
        ) = None
        input_68 = input_59 + input_67
        input_59 = input_67 = None
        input_69 = torch.conv2d(
            input_68,
            l_self_modules_features_modules_9_modules_conv_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_9_modules_conv_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_70 = torch.nn.functional.batch_norm(
            input_69,
            l_self_modules_features_modules_9_modules_conv_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_9_modules_conv_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_9_modules_conv_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_9_modules_conv_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_69 = l_self_modules_features_modules_9_modules_conv_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_9_modules_conv_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_9_modules_conv_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_9_modules_conv_modules_0_modules_1_parameters_bias_ = (None)
        input_71 = torch.nn.functional.hardtanh(input_70, 0.0, 6.0, True)
        input_70 = None
        input_72 = torch.conv2d(
            input_71,
            l_self_modules_features_modules_9_modules_conv_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        input_71 = l_self_modules_features_modules_9_modules_conv_modules_1_modules_0_parameters_weight_ = (None)
        input_73 = torch.nn.functional.batch_norm(
            input_72,
            l_self_modules_features_modules_9_modules_conv_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_9_modules_conv_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_9_modules_conv_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_9_modules_conv_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_72 = l_self_modules_features_modules_9_modules_conv_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_9_modules_conv_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_9_modules_conv_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_9_modules_conv_modules_1_modules_1_parameters_bias_ = (None)
        input_74 = torch.nn.functional.hardtanh(input_73, 0.0, 6.0, True)
        input_73 = None
        input_75 = torch.conv2d(
            input_74,
            l_self_modules_features_modules_9_modules_conv_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_74 = (
            l_self_modules_features_modules_9_modules_conv_modules_2_parameters_weight_
        ) = None
        input_76 = torch.nn.functional.batch_norm(
            input_75,
            l_self_modules_features_modules_9_modules_conv_modules_3_buffers_running_mean_,
            l_self_modules_features_modules_9_modules_conv_modules_3_buffers_running_var_,
            l_self_modules_features_modules_9_modules_conv_modules_3_parameters_weight_,
            l_self_modules_features_modules_9_modules_conv_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_75 = l_self_modules_features_modules_9_modules_conv_modules_3_buffers_running_mean_ = l_self_modules_features_modules_9_modules_conv_modules_3_buffers_running_var_ = (
            l_self_modules_features_modules_9_modules_conv_modules_3_parameters_weight_
        ) = (
            l_self_modules_features_modules_9_modules_conv_modules_3_parameters_bias_
        ) = None
        input_77 = input_68 + input_76
        input_68 = input_76 = None
        input_78 = torch.conv2d(
            input_77,
            l_self_modules_features_modules_10_modules_conv_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_10_modules_conv_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_79 = torch.nn.functional.batch_norm(
            input_78,
            l_self_modules_features_modules_10_modules_conv_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_10_modules_conv_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_10_modules_conv_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_10_modules_conv_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_78 = l_self_modules_features_modules_10_modules_conv_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_10_modules_conv_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_10_modules_conv_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_10_modules_conv_modules_0_modules_1_parameters_bias_ = (None)
        input_80 = torch.nn.functional.hardtanh(input_79, 0.0, 6.0, True)
        input_79 = None
        input_81 = torch.conv2d(
            input_80,
            l_self_modules_features_modules_10_modules_conv_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        input_80 = l_self_modules_features_modules_10_modules_conv_modules_1_modules_0_parameters_weight_ = (None)
        input_82 = torch.nn.functional.batch_norm(
            input_81,
            l_self_modules_features_modules_10_modules_conv_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_10_modules_conv_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_10_modules_conv_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_10_modules_conv_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_81 = l_self_modules_features_modules_10_modules_conv_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_10_modules_conv_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_10_modules_conv_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_10_modules_conv_modules_1_modules_1_parameters_bias_ = (None)
        input_83 = torch.nn.functional.hardtanh(input_82, 0.0, 6.0, True)
        input_82 = None
        input_84 = torch.conv2d(
            input_83,
            l_self_modules_features_modules_10_modules_conv_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_83 = (
            l_self_modules_features_modules_10_modules_conv_modules_2_parameters_weight_
        ) = None
        input_85 = torch.nn.functional.batch_norm(
            input_84,
            l_self_modules_features_modules_10_modules_conv_modules_3_buffers_running_mean_,
            l_self_modules_features_modules_10_modules_conv_modules_3_buffers_running_var_,
            l_self_modules_features_modules_10_modules_conv_modules_3_parameters_weight_,
            l_self_modules_features_modules_10_modules_conv_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_84 = l_self_modules_features_modules_10_modules_conv_modules_3_buffers_running_mean_ = l_self_modules_features_modules_10_modules_conv_modules_3_buffers_running_var_ = (
            l_self_modules_features_modules_10_modules_conv_modules_3_parameters_weight_
        ) = (
            l_self_modules_features_modules_10_modules_conv_modules_3_parameters_bias_
        ) = None
        input_86 = input_77 + input_85
        input_77 = input_85 = None
        input_87 = torch.conv2d(
            input_86,
            l_self_modules_features_modules_11_modules_conv_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_86 = l_self_modules_features_modules_11_modules_conv_modules_0_modules_0_parameters_weight_ = (None)
        input_88 = torch.nn.functional.batch_norm(
            input_87,
            l_self_modules_features_modules_11_modules_conv_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_conv_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_11_modules_conv_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_11_modules_conv_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_87 = l_self_modules_features_modules_11_modules_conv_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_11_modules_conv_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_11_modules_conv_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_11_modules_conv_modules_0_modules_1_parameters_bias_ = (None)
        input_89 = torch.nn.functional.hardtanh(input_88, 0.0, 6.0, True)
        input_88 = None
        input_90 = torch.conv2d(
            input_89,
            l_self_modules_features_modules_11_modules_conv_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            384,
        )
        input_89 = l_self_modules_features_modules_11_modules_conv_modules_1_modules_0_parameters_weight_ = (None)
        input_91 = torch.nn.functional.batch_norm(
            input_90,
            l_self_modules_features_modules_11_modules_conv_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_conv_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_11_modules_conv_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_11_modules_conv_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_90 = l_self_modules_features_modules_11_modules_conv_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_11_modules_conv_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_11_modules_conv_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_11_modules_conv_modules_1_modules_1_parameters_bias_ = (None)
        input_92 = torch.nn.functional.hardtanh(input_91, 0.0, 6.0, True)
        input_91 = None
        input_93 = torch.conv2d(
            input_92,
            l_self_modules_features_modules_11_modules_conv_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_92 = (
            l_self_modules_features_modules_11_modules_conv_modules_2_parameters_weight_
        ) = None
        input_94 = torch.nn.functional.batch_norm(
            input_93,
            l_self_modules_features_modules_11_modules_conv_modules_3_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_conv_modules_3_buffers_running_var_,
            l_self_modules_features_modules_11_modules_conv_modules_3_parameters_weight_,
            l_self_modules_features_modules_11_modules_conv_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_93 = l_self_modules_features_modules_11_modules_conv_modules_3_buffers_running_mean_ = l_self_modules_features_modules_11_modules_conv_modules_3_buffers_running_var_ = (
            l_self_modules_features_modules_11_modules_conv_modules_3_parameters_weight_
        ) = (
            l_self_modules_features_modules_11_modules_conv_modules_3_parameters_bias_
        ) = None
        input_95 = torch.conv2d(
            input_94,
            l_self_modules_features_modules_12_modules_conv_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_12_modules_conv_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_96 = torch.nn.functional.batch_norm(
            input_95,
            l_self_modules_features_modules_12_modules_conv_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_12_modules_conv_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_12_modules_conv_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_12_modules_conv_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_95 = l_self_modules_features_modules_12_modules_conv_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_12_modules_conv_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_12_modules_conv_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_12_modules_conv_modules_0_modules_1_parameters_bias_ = (None)
        input_97 = torch.nn.functional.hardtanh(input_96, 0.0, 6.0, True)
        input_96 = None
        input_98 = torch.conv2d(
            input_97,
            l_self_modules_features_modules_12_modules_conv_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            576,
        )
        input_97 = l_self_modules_features_modules_12_modules_conv_modules_1_modules_0_parameters_weight_ = (None)
        input_99 = torch.nn.functional.batch_norm(
            input_98,
            l_self_modules_features_modules_12_modules_conv_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_12_modules_conv_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_12_modules_conv_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_12_modules_conv_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_98 = l_self_modules_features_modules_12_modules_conv_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_12_modules_conv_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_12_modules_conv_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_12_modules_conv_modules_1_modules_1_parameters_bias_ = (None)
        input_100 = torch.nn.functional.hardtanh(input_99, 0.0, 6.0, True)
        input_99 = None
        input_101 = torch.conv2d(
            input_100,
            l_self_modules_features_modules_12_modules_conv_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_100 = (
            l_self_modules_features_modules_12_modules_conv_modules_2_parameters_weight_
        ) = None
        input_102 = torch.nn.functional.batch_norm(
            input_101,
            l_self_modules_features_modules_12_modules_conv_modules_3_buffers_running_mean_,
            l_self_modules_features_modules_12_modules_conv_modules_3_buffers_running_var_,
            l_self_modules_features_modules_12_modules_conv_modules_3_parameters_weight_,
            l_self_modules_features_modules_12_modules_conv_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_101 = l_self_modules_features_modules_12_modules_conv_modules_3_buffers_running_mean_ = l_self_modules_features_modules_12_modules_conv_modules_3_buffers_running_var_ = (
            l_self_modules_features_modules_12_modules_conv_modules_3_parameters_weight_
        ) = (
            l_self_modules_features_modules_12_modules_conv_modules_3_parameters_bias_
        ) = None
        input_103 = input_94 + input_102
        input_94 = input_102 = None
        input_104 = torch.conv2d(
            input_103,
            l_self_modules_features_modules_13_modules_conv_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_13_modules_conv_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_105 = torch.nn.functional.batch_norm(
            input_104,
            l_self_modules_features_modules_13_modules_conv_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_13_modules_conv_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_13_modules_conv_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_13_modules_conv_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_104 = l_self_modules_features_modules_13_modules_conv_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_13_modules_conv_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_13_modules_conv_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_13_modules_conv_modules_0_modules_1_parameters_bias_ = (None)
        input_106 = torch.nn.functional.hardtanh(input_105, 0.0, 6.0, True)
        input_105 = None
        input_107 = torch.conv2d(
            input_106,
            l_self_modules_features_modules_13_modules_conv_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            576,
        )
        input_106 = l_self_modules_features_modules_13_modules_conv_modules_1_modules_0_parameters_weight_ = (None)
        input_108 = torch.nn.functional.batch_norm(
            input_107,
            l_self_modules_features_modules_13_modules_conv_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_13_modules_conv_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_13_modules_conv_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_13_modules_conv_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_107 = l_self_modules_features_modules_13_modules_conv_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_13_modules_conv_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_13_modules_conv_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_13_modules_conv_modules_1_modules_1_parameters_bias_ = (None)
        input_109 = torch.nn.functional.hardtanh(input_108, 0.0, 6.0, True)
        input_108 = None
        input_110 = torch.conv2d(
            input_109,
            l_self_modules_features_modules_13_modules_conv_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_109 = (
            l_self_modules_features_modules_13_modules_conv_modules_2_parameters_weight_
        ) = None
        input_111 = torch.nn.functional.batch_norm(
            input_110,
            l_self_modules_features_modules_13_modules_conv_modules_3_buffers_running_mean_,
            l_self_modules_features_modules_13_modules_conv_modules_3_buffers_running_var_,
            l_self_modules_features_modules_13_modules_conv_modules_3_parameters_weight_,
            l_self_modules_features_modules_13_modules_conv_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_110 = l_self_modules_features_modules_13_modules_conv_modules_3_buffers_running_mean_ = l_self_modules_features_modules_13_modules_conv_modules_3_buffers_running_var_ = (
            l_self_modules_features_modules_13_modules_conv_modules_3_parameters_weight_
        ) = (
            l_self_modules_features_modules_13_modules_conv_modules_3_parameters_bias_
        ) = None
        input_112 = input_103 + input_111
        input_103 = input_111 = None
        input_113 = torch.conv2d(
            input_112,
            l_self_modules_features_modules_14_modules_conv_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_112 = l_self_modules_features_modules_14_modules_conv_modules_0_modules_0_parameters_weight_ = (None)
        input_114 = torch.nn.functional.batch_norm(
            input_113,
            l_self_modules_features_modules_14_modules_conv_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_14_modules_conv_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_14_modules_conv_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_14_modules_conv_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_113 = l_self_modules_features_modules_14_modules_conv_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_14_modules_conv_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_14_modules_conv_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_14_modules_conv_modules_0_modules_1_parameters_bias_ = (None)
        input_115 = torch.nn.functional.hardtanh(input_114, 0.0, 6.0, True)
        input_114 = None
        input_116 = torch.conv2d(
            input_115,
            l_self_modules_features_modules_14_modules_conv_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            576,
        )
        input_115 = l_self_modules_features_modules_14_modules_conv_modules_1_modules_0_parameters_weight_ = (None)
        input_117 = torch.nn.functional.batch_norm(
            input_116,
            l_self_modules_features_modules_14_modules_conv_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_14_modules_conv_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_14_modules_conv_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_14_modules_conv_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_116 = l_self_modules_features_modules_14_modules_conv_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_14_modules_conv_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_14_modules_conv_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_14_modules_conv_modules_1_modules_1_parameters_bias_ = (None)
        input_118 = torch.nn.functional.hardtanh(input_117, 0.0, 6.0, True)
        input_117 = None
        input_119 = torch.conv2d(
            input_118,
            l_self_modules_features_modules_14_modules_conv_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_118 = (
            l_self_modules_features_modules_14_modules_conv_modules_2_parameters_weight_
        ) = None
        input_120 = torch.nn.functional.batch_norm(
            input_119,
            l_self_modules_features_modules_14_modules_conv_modules_3_buffers_running_mean_,
            l_self_modules_features_modules_14_modules_conv_modules_3_buffers_running_var_,
            l_self_modules_features_modules_14_modules_conv_modules_3_parameters_weight_,
            l_self_modules_features_modules_14_modules_conv_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_119 = l_self_modules_features_modules_14_modules_conv_modules_3_buffers_running_mean_ = l_self_modules_features_modules_14_modules_conv_modules_3_buffers_running_var_ = (
            l_self_modules_features_modules_14_modules_conv_modules_3_parameters_weight_
        ) = (
            l_self_modules_features_modules_14_modules_conv_modules_3_parameters_bias_
        ) = None
        input_121 = torch.conv2d(
            input_120,
            l_self_modules_features_modules_15_modules_conv_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_15_modules_conv_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_122 = torch.nn.functional.batch_norm(
            input_121,
            l_self_modules_features_modules_15_modules_conv_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_15_modules_conv_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_15_modules_conv_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_15_modules_conv_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_121 = l_self_modules_features_modules_15_modules_conv_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_15_modules_conv_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_15_modules_conv_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_15_modules_conv_modules_0_modules_1_parameters_bias_ = (None)
        input_123 = torch.nn.functional.hardtanh(input_122, 0.0, 6.0, True)
        input_122 = None
        input_124 = torch.conv2d(
            input_123,
            l_self_modules_features_modules_15_modules_conv_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_123 = l_self_modules_features_modules_15_modules_conv_modules_1_modules_0_parameters_weight_ = (None)
        input_125 = torch.nn.functional.batch_norm(
            input_124,
            l_self_modules_features_modules_15_modules_conv_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_15_modules_conv_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_15_modules_conv_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_15_modules_conv_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_124 = l_self_modules_features_modules_15_modules_conv_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_15_modules_conv_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_15_modules_conv_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_15_modules_conv_modules_1_modules_1_parameters_bias_ = (None)
        input_126 = torch.nn.functional.hardtanh(input_125, 0.0, 6.0, True)
        input_125 = None
        input_127 = torch.conv2d(
            input_126,
            l_self_modules_features_modules_15_modules_conv_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_126 = (
            l_self_modules_features_modules_15_modules_conv_modules_2_parameters_weight_
        ) = None
        input_128 = torch.nn.functional.batch_norm(
            input_127,
            l_self_modules_features_modules_15_modules_conv_modules_3_buffers_running_mean_,
            l_self_modules_features_modules_15_modules_conv_modules_3_buffers_running_var_,
            l_self_modules_features_modules_15_modules_conv_modules_3_parameters_weight_,
            l_self_modules_features_modules_15_modules_conv_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_127 = l_self_modules_features_modules_15_modules_conv_modules_3_buffers_running_mean_ = l_self_modules_features_modules_15_modules_conv_modules_3_buffers_running_var_ = (
            l_self_modules_features_modules_15_modules_conv_modules_3_parameters_weight_
        ) = (
            l_self_modules_features_modules_15_modules_conv_modules_3_parameters_bias_
        ) = None
        input_129 = input_120 + input_128
        input_120 = input_128 = None
        input_130 = torch.conv2d(
            input_129,
            l_self_modules_features_modules_16_modules_conv_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_16_modules_conv_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_131 = torch.nn.functional.batch_norm(
            input_130,
            l_self_modules_features_modules_16_modules_conv_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_16_modules_conv_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_16_modules_conv_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_16_modules_conv_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_130 = l_self_modules_features_modules_16_modules_conv_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_16_modules_conv_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_16_modules_conv_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_16_modules_conv_modules_0_modules_1_parameters_bias_ = (None)
        input_132 = torch.nn.functional.hardtanh(input_131, 0.0, 6.0, True)
        input_131 = None
        input_133 = torch.conv2d(
            input_132,
            l_self_modules_features_modules_16_modules_conv_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_132 = l_self_modules_features_modules_16_modules_conv_modules_1_modules_0_parameters_weight_ = (None)
        input_134 = torch.nn.functional.batch_norm(
            input_133,
            l_self_modules_features_modules_16_modules_conv_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_16_modules_conv_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_16_modules_conv_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_16_modules_conv_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_133 = l_self_modules_features_modules_16_modules_conv_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_16_modules_conv_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_16_modules_conv_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_16_modules_conv_modules_1_modules_1_parameters_bias_ = (None)
        input_135 = torch.nn.functional.hardtanh(input_134, 0.0, 6.0, True)
        input_134 = None
        input_136 = torch.conv2d(
            input_135,
            l_self_modules_features_modules_16_modules_conv_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_135 = (
            l_self_modules_features_modules_16_modules_conv_modules_2_parameters_weight_
        ) = None
        input_137 = torch.nn.functional.batch_norm(
            input_136,
            l_self_modules_features_modules_16_modules_conv_modules_3_buffers_running_mean_,
            l_self_modules_features_modules_16_modules_conv_modules_3_buffers_running_var_,
            l_self_modules_features_modules_16_modules_conv_modules_3_parameters_weight_,
            l_self_modules_features_modules_16_modules_conv_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_136 = l_self_modules_features_modules_16_modules_conv_modules_3_buffers_running_mean_ = l_self_modules_features_modules_16_modules_conv_modules_3_buffers_running_var_ = (
            l_self_modules_features_modules_16_modules_conv_modules_3_parameters_weight_
        ) = (
            l_self_modules_features_modules_16_modules_conv_modules_3_parameters_bias_
        ) = None
        input_138 = input_129 + input_137
        input_129 = input_137 = None
        input_139 = torch.conv2d(
            input_138,
            l_self_modules_features_modules_17_modules_conv_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_138 = l_self_modules_features_modules_17_modules_conv_modules_0_modules_0_parameters_weight_ = (None)
        input_140 = torch.nn.functional.batch_norm(
            input_139,
            l_self_modules_features_modules_17_modules_conv_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_17_modules_conv_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_17_modules_conv_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_17_modules_conv_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_139 = l_self_modules_features_modules_17_modules_conv_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_17_modules_conv_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_17_modules_conv_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_17_modules_conv_modules_0_modules_1_parameters_bias_ = (None)
        input_141 = torch.nn.functional.hardtanh(input_140, 0.0, 6.0, True)
        input_140 = None
        input_142 = torch.conv2d(
            input_141,
            l_self_modules_features_modules_17_modules_conv_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_141 = l_self_modules_features_modules_17_modules_conv_modules_1_modules_0_parameters_weight_ = (None)
        input_143 = torch.nn.functional.batch_norm(
            input_142,
            l_self_modules_features_modules_17_modules_conv_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_17_modules_conv_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_17_modules_conv_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_17_modules_conv_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_142 = l_self_modules_features_modules_17_modules_conv_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_17_modules_conv_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_17_modules_conv_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_17_modules_conv_modules_1_modules_1_parameters_bias_ = (None)
        input_144 = torch.nn.functional.hardtanh(input_143, 0.0, 6.0, True)
        input_143 = None
        input_145 = torch.conv2d(
            input_144,
            l_self_modules_features_modules_17_modules_conv_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_144 = (
            l_self_modules_features_modules_17_modules_conv_modules_2_parameters_weight_
        ) = None
        input_146 = torch.nn.functional.batch_norm(
            input_145,
            l_self_modules_features_modules_17_modules_conv_modules_3_buffers_running_mean_,
            l_self_modules_features_modules_17_modules_conv_modules_3_buffers_running_var_,
            l_self_modules_features_modules_17_modules_conv_modules_3_parameters_weight_,
            l_self_modules_features_modules_17_modules_conv_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_145 = l_self_modules_features_modules_17_modules_conv_modules_3_buffers_running_mean_ = l_self_modules_features_modules_17_modules_conv_modules_3_buffers_running_var_ = (
            l_self_modules_features_modules_17_modules_conv_modules_3_parameters_weight_
        ) = (
            l_self_modules_features_modules_17_modules_conv_modules_3_parameters_bias_
        ) = None
        input_147 = torch.conv2d(
            input_146,
            l_self_modules_features_modules_18_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_146 = (
            l_self_modules_features_modules_18_modules_0_parameters_weight_
        ) = None
        input_148 = torch.nn.functional.batch_norm(
            input_147,
            l_self_modules_features_modules_18_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_18_modules_1_buffers_running_var_,
            l_self_modules_features_modules_18_modules_1_parameters_weight_,
            l_self_modules_features_modules_18_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_147 = (
            l_self_modules_features_modules_18_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_18_modules_1_buffers_running_var_
        ) = (
            l_self_modules_features_modules_18_modules_1_parameters_weight_
        ) = l_self_modules_features_modules_18_modules_1_parameters_bias_ = None
        input_149 = torch.nn.functional.hardtanh(input_148, 0.0, 6.0, True)
        input_148 = None
        x = torch.nn.functional.adaptive_avg_pool2d(input_149, (1, 1))
        input_149 = None
        x_1 = torch.flatten(x, 1)
        x = None
        input_150 = torch.nn.functional.dropout(x_1, 0.2, False, False)
        x_1 = None
        input_151 = torch._C._nn.linear(
            input_150,
            l_self_modules_classifier_modules_1_parameters_weight_,
            l_self_modules_classifier_modules_1_parameters_bias_,
        )
        input_150 = (
            l_self_modules_classifier_modules_1_parameters_weight_
        ) = l_self_modules_classifier_modules_1_parameters_bias_ = None
        return (input_151,)
