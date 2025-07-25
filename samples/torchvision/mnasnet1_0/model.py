import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_0_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_0_modules_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_0_modules_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_1_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_1_modules_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_1_modules_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_2_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_2_modules_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_2_modules_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_0_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_0_modules_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_0_modules_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_1_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_1_modules_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_1_modules_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_2_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_2_modules_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_2_modules_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_0_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_0_modules_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_0_modules_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_1_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_1_modules_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_1_modules_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_2_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_2_modules_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_2_modules_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_0_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_0_modules_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_0_modules_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_1_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_1_modules_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_1_modules_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_0_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_0_modules_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_0_modules_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_1_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_1_modules_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_1_modules_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_2_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_2_modules_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_2_modules_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_3_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_3_modules_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_3_modules_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_0_modules_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_0_modules_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_0_modules_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_14_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_buffers_running_mean_: torch.Tensor,
        L_self_modules_layers_modules_15_buffers_running_var_: torch.Tensor,
        L_self_modules_layers_modules_15_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layers_modules_15_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_layers_modules_0_parameters_weight_ = (
            L_self_modules_layers_modules_0_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_layers_modules_1_buffers_running_mean_ = (
            L_self_modules_layers_modules_1_buffers_running_mean_
        )
        l_self_modules_layers_modules_1_buffers_running_var_ = (
            L_self_modules_layers_modules_1_buffers_running_var_
        )
        l_self_modules_layers_modules_1_parameters_weight_ = (
            L_self_modules_layers_modules_1_parameters_weight_
        )
        l_self_modules_layers_modules_1_parameters_bias_ = (
            L_self_modules_layers_modules_1_parameters_bias_
        )
        l_self_modules_layers_modules_3_parameters_weight_ = (
            L_self_modules_layers_modules_3_parameters_weight_
        )
        l_self_modules_layers_modules_4_buffers_running_mean_ = (
            L_self_modules_layers_modules_4_buffers_running_mean_
        )
        l_self_modules_layers_modules_4_buffers_running_var_ = (
            L_self_modules_layers_modules_4_buffers_running_var_
        )
        l_self_modules_layers_modules_4_parameters_weight_ = (
            L_self_modules_layers_modules_4_parameters_weight_
        )
        l_self_modules_layers_modules_4_parameters_bias_ = (
            L_self_modules_layers_modules_4_parameters_bias_
        )
        l_self_modules_layers_modules_6_parameters_weight_ = (
            L_self_modules_layers_modules_6_parameters_weight_
        )
        l_self_modules_layers_modules_7_buffers_running_mean_ = (
            L_self_modules_layers_modules_7_buffers_running_mean_
        )
        l_self_modules_layers_modules_7_buffers_running_var_ = (
            L_self_modules_layers_modules_7_buffers_running_var_
        )
        l_self_modules_layers_modules_7_parameters_weight_ = (
            L_self_modules_layers_modules_7_parameters_weight_
        )
        l_self_modules_layers_modules_7_parameters_bias_ = (
            L_self_modules_layers_modules_7_parameters_bias_
        )
        l_self_modules_layers_modules_8_modules_0_modules_layers_modules_0_parameters_weight_ = L_self_modules_layers_modules_8_modules_0_modules_layers_modules_0_parameters_weight_
        l_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_buffers_running_mean_ = L_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_buffers_running_mean_
        l_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_buffers_running_var_ = L_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_buffers_running_var_
        l_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_parameters_weight_ = L_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_parameters_weight_
        l_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_parameters_bias_ = L_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_parameters_bias_
        l_self_modules_layers_modules_8_modules_0_modules_layers_modules_3_parameters_weight_ = L_self_modules_layers_modules_8_modules_0_modules_layers_modules_3_parameters_weight_
        l_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_buffers_running_mean_ = L_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_buffers_running_mean_
        l_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_buffers_running_var_ = L_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_buffers_running_var_
        l_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_parameters_weight_ = L_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_parameters_weight_
        l_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_parameters_bias_ = L_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_parameters_bias_
        l_self_modules_layers_modules_8_modules_0_modules_layers_modules_6_parameters_weight_ = L_self_modules_layers_modules_8_modules_0_modules_layers_modules_6_parameters_weight_
        l_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_buffers_running_mean_ = L_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_buffers_running_mean_
        l_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_buffers_running_var_ = L_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_buffers_running_var_
        l_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_parameters_weight_ = L_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_parameters_weight_
        l_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_parameters_bias_ = L_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_parameters_bias_
        l_self_modules_layers_modules_8_modules_1_modules_layers_modules_0_parameters_weight_ = L_self_modules_layers_modules_8_modules_1_modules_layers_modules_0_parameters_weight_
        l_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_buffers_running_mean_ = L_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_buffers_running_mean_
        l_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_buffers_running_var_ = L_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_buffers_running_var_
        l_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_parameters_weight_ = L_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_parameters_weight_
        l_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_parameters_bias_ = L_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_parameters_bias_
        l_self_modules_layers_modules_8_modules_1_modules_layers_modules_3_parameters_weight_ = L_self_modules_layers_modules_8_modules_1_modules_layers_modules_3_parameters_weight_
        l_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_buffers_running_mean_ = L_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_buffers_running_mean_
        l_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_buffers_running_var_ = L_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_buffers_running_var_
        l_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_parameters_weight_ = L_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_parameters_weight_
        l_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_parameters_bias_ = L_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_parameters_bias_
        l_self_modules_layers_modules_8_modules_1_modules_layers_modules_6_parameters_weight_ = L_self_modules_layers_modules_8_modules_1_modules_layers_modules_6_parameters_weight_
        l_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_buffers_running_mean_ = L_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_buffers_running_mean_
        l_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_buffers_running_var_ = L_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_buffers_running_var_
        l_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_parameters_weight_ = L_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_parameters_weight_
        l_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_parameters_bias_ = L_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_parameters_bias_
        l_self_modules_layers_modules_8_modules_2_modules_layers_modules_0_parameters_weight_ = L_self_modules_layers_modules_8_modules_2_modules_layers_modules_0_parameters_weight_
        l_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_buffers_running_mean_ = L_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_buffers_running_mean_
        l_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_buffers_running_var_ = L_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_buffers_running_var_
        l_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_parameters_weight_ = L_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_parameters_weight_
        l_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_parameters_bias_ = L_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_parameters_bias_
        l_self_modules_layers_modules_8_modules_2_modules_layers_modules_3_parameters_weight_ = L_self_modules_layers_modules_8_modules_2_modules_layers_modules_3_parameters_weight_
        l_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_buffers_running_mean_ = L_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_buffers_running_mean_
        l_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_buffers_running_var_ = L_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_buffers_running_var_
        l_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_parameters_weight_ = L_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_parameters_weight_
        l_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_parameters_bias_ = L_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_parameters_bias_
        l_self_modules_layers_modules_8_modules_2_modules_layers_modules_6_parameters_weight_ = L_self_modules_layers_modules_8_modules_2_modules_layers_modules_6_parameters_weight_
        l_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_buffers_running_mean_ = L_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_buffers_running_mean_
        l_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_buffers_running_var_ = L_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_buffers_running_var_
        l_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_parameters_weight_ = L_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_parameters_weight_
        l_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_parameters_bias_ = L_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_parameters_bias_
        l_self_modules_layers_modules_9_modules_0_modules_layers_modules_0_parameters_weight_ = L_self_modules_layers_modules_9_modules_0_modules_layers_modules_0_parameters_weight_
        l_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_buffers_running_mean_ = L_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_buffers_running_mean_
        l_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_buffers_running_var_ = L_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_buffers_running_var_
        l_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_parameters_weight_ = L_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_parameters_weight_
        l_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_parameters_bias_ = L_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_parameters_bias_
        l_self_modules_layers_modules_9_modules_0_modules_layers_modules_3_parameters_weight_ = L_self_modules_layers_modules_9_modules_0_modules_layers_modules_3_parameters_weight_
        l_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_buffers_running_mean_ = L_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_buffers_running_mean_
        l_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_buffers_running_var_ = L_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_buffers_running_var_
        l_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_parameters_weight_ = L_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_parameters_weight_
        l_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_parameters_bias_ = L_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_parameters_bias_
        l_self_modules_layers_modules_9_modules_0_modules_layers_modules_6_parameters_weight_ = L_self_modules_layers_modules_9_modules_0_modules_layers_modules_6_parameters_weight_
        l_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_buffers_running_mean_ = L_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_buffers_running_mean_
        l_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_buffers_running_var_ = L_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_buffers_running_var_
        l_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_parameters_weight_ = L_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_parameters_weight_
        l_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_parameters_bias_ = L_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_parameters_bias_
        l_self_modules_layers_modules_9_modules_1_modules_layers_modules_0_parameters_weight_ = L_self_modules_layers_modules_9_modules_1_modules_layers_modules_0_parameters_weight_
        l_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_buffers_running_mean_ = L_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_buffers_running_mean_
        l_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_buffers_running_var_ = L_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_buffers_running_var_
        l_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_parameters_weight_ = L_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_parameters_weight_
        l_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_parameters_bias_ = L_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_parameters_bias_
        l_self_modules_layers_modules_9_modules_1_modules_layers_modules_3_parameters_weight_ = L_self_modules_layers_modules_9_modules_1_modules_layers_modules_3_parameters_weight_
        l_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_buffers_running_mean_ = L_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_buffers_running_mean_
        l_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_buffers_running_var_ = L_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_buffers_running_var_
        l_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_parameters_weight_ = L_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_parameters_weight_
        l_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_parameters_bias_ = L_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_parameters_bias_
        l_self_modules_layers_modules_9_modules_1_modules_layers_modules_6_parameters_weight_ = L_self_modules_layers_modules_9_modules_1_modules_layers_modules_6_parameters_weight_
        l_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_buffers_running_mean_ = L_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_buffers_running_mean_
        l_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_buffers_running_var_ = L_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_buffers_running_var_
        l_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_parameters_weight_ = L_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_parameters_weight_
        l_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_parameters_bias_ = L_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_parameters_bias_
        l_self_modules_layers_modules_9_modules_2_modules_layers_modules_0_parameters_weight_ = L_self_modules_layers_modules_9_modules_2_modules_layers_modules_0_parameters_weight_
        l_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_buffers_running_mean_ = L_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_buffers_running_mean_
        l_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_buffers_running_var_ = L_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_buffers_running_var_
        l_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_parameters_weight_ = L_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_parameters_weight_
        l_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_parameters_bias_ = L_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_parameters_bias_
        l_self_modules_layers_modules_9_modules_2_modules_layers_modules_3_parameters_weight_ = L_self_modules_layers_modules_9_modules_2_modules_layers_modules_3_parameters_weight_
        l_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_buffers_running_mean_ = L_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_buffers_running_mean_
        l_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_buffers_running_var_ = L_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_buffers_running_var_
        l_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_parameters_weight_ = L_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_parameters_weight_
        l_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_parameters_bias_ = L_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_parameters_bias_
        l_self_modules_layers_modules_9_modules_2_modules_layers_modules_6_parameters_weight_ = L_self_modules_layers_modules_9_modules_2_modules_layers_modules_6_parameters_weight_
        l_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_buffers_running_mean_ = L_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_buffers_running_mean_
        l_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_buffers_running_var_ = L_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_buffers_running_var_
        l_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_parameters_weight_ = L_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_parameters_weight_
        l_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_parameters_bias_ = L_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_parameters_bias_
        l_self_modules_layers_modules_10_modules_0_modules_layers_modules_0_parameters_weight_ = L_self_modules_layers_modules_10_modules_0_modules_layers_modules_0_parameters_weight_
        l_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_buffers_running_mean_ = L_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_buffers_running_mean_
        l_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_buffers_running_var_ = L_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_buffers_running_var_
        l_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_parameters_weight_ = L_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_parameters_weight_
        l_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_parameters_bias_ = L_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_parameters_bias_
        l_self_modules_layers_modules_10_modules_0_modules_layers_modules_3_parameters_weight_ = L_self_modules_layers_modules_10_modules_0_modules_layers_modules_3_parameters_weight_
        l_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_buffers_running_mean_ = L_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_buffers_running_mean_
        l_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_buffers_running_var_ = L_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_buffers_running_var_
        l_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_parameters_weight_ = L_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_parameters_weight_
        l_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_parameters_bias_ = L_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_parameters_bias_
        l_self_modules_layers_modules_10_modules_0_modules_layers_modules_6_parameters_weight_ = L_self_modules_layers_modules_10_modules_0_modules_layers_modules_6_parameters_weight_
        l_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_buffers_running_mean_ = L_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_buffers_running_mean_
        l_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_buffers_running_var_ = L_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_buffers_running_var_
        l_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_parameters_weight_ = L_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_parameters_weight_
        l_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_parameters_bias_ = L_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_parameters_bias_
        l_self_modules_layers_modules_10_modules_1_modules_layers_modules_0_parameters_weight_ = L_self_modules_layers_modules_10_modules_1_modules_layers_modules_0_parameters_weight_
        l_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_buffers_running_mean_ = L_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_buffers_running_mean_
        l_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_buffers_running_var_ = L_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_buffers_running_var_
        l_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_parameters_weight_ = L_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_parameters_weight_
        l_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_parameters_bias_ = L_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_parameters_bias_
        l_self_modules_layers_modules_10_modules_1_modules_layers_modules_3_parameters_weight_ = L_self_modules_layers_modules_10_modules_1_modules_layers_modules_3_parameters_weight_
        l_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_buffers_running_mean_ = L_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_buffers_running_mean_
        l_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_buffers_running_var_ = L_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_buffers_running_var_
        l_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_parameters_weight_ = L_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_parameters_weight_
        l_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_parameters_bias_ = L_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_parameters_bias_
        l_self_modules_layers_modules_10_modules_1_modules_layers_modules_6_parameters_weight_ = L_self_modules_layers_modules_10_modules_1_modules_layers_modules_6_parameters_weight_
        l_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_buffers_running_mean_ = L_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_buffers_running_mean_
        l_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_buffers_running_var_ = L_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_buffers_running_var_
        l_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_parameters_weight_ = L_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_parameters_weight_
        l_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_parameters_bias_ = L_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_parameters_bias_
        l_self_modules_layers_modules_10_modules_2_modules_layers_modules_0_parameters_weight_ = L_self_modules_layers_modules_10_modules_2_modules_layers_modules_0_parameters_weight_
        l_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_buffers_running_mean_ = L_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_buffers_running_mean_
        l_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_buffers_running_var_ = L_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_buffers_running_var_
        l_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_parameters_weight_ = L_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_parameters_weight_
        l_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_parameters_bias_ = L_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_parameters_bias_
        l_self_modules_layers_modules_10_modules_2_modules_layers_modules_3_parameters_weight_ = L_self_modules_layers_modules_10_modules_2_modules_layers_modules_3_parameters_weight_
        l_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_buffers_running_mean_ = L_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_buffers_running_mean_
        l_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_buffers_running_var_ = L_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_buffers_running_var_
        l_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_parameters_weight_ = L_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_parameters_weight_
        l_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_parameters_bias_ = L_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_parameters_bias_
        l_self_modules_layers_modules_10_modules_2_modules_layers_modules_6_parameters_weight_ = L_self_modules_layers_modules_10_modules_2_modules_layers_modules_6_parameters_weight_
        l_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_buffers_running_mean_ = L_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_buffers_running_mean_
        l_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_buffers_running_var_ = L_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_buffers_running_var_
        l_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_parameters_weight_ = L_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_parameters_weight_
        l_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_parameters_bias_ = L_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_parameters_bias_
        l_self_modules_layers_modules_11_modules_0_modules_layers_modules_0_parameters_weight_ = L_self_modules_layers_modules_11_modules_0_modules_layers_modules_0_parameters_weight_
        l_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_buffers_running_mean_ = L_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_buffers_running_mean_
        l_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_buffers_running_var_ = L_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_buffers_running_var_
        l_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_parameters_weight_ = L_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_parameters_weight_
        l_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_parameters_bias_ = L_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_parameters_bias_
        l_self_modules_layers_modules_11_modules_0_modules_layers_modules_3_parameters_weight_ = L_self_modules_layers_modules_11_modules_0_modules_layers_modules_3_parameters_weight_
        l_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_buffers_running_mean_ = L_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_buffers_running_mean_
        l_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_buffers_running_var_ = L_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_buffers_running_var_
        l_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_parameters_weight_ = L_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_parameters_weight_
        l_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_parameters_bias_ = L_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_parameters_bias_
        l_self_modules_layers_modules_11_modules_0_modules_layers_modules_6_parameters_weight_ = L_self_modules_layers_modules_11_modules_0_modules_layers_modules_6_parameters_weight_
        l_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_buffers_running_mean_ = L_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_buffers_running_mean_
        l_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_buffers_running_var_ = L_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_buffers_running_var_
        l_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_parameters_weight_ = L_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_parameters_weight_
        l_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_parameters_bias_ = L_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_parameters_bias_
        l_self_modules_layers_modules_11_modules_1_modules_layers_modules_0_parameters_weight_ = L_self_modules_layers_modules_11_modules_1_modules_layers_modules_0_parameters_weight_
        l_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_buffers_running_mean_ = L_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_buffers_running_mean_
        l_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_buffers_running_var_ = L_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_buffers_running_var_
        l_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_parameters_weight_ = L_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_parameters_weight_
        l_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_parameters_bias_ = L_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_parameters_bias_
        l_self_modules_layers_modules_11_modules_1_modules_layers_modules_3_parameters_weight_ = L_self_modules_layers_modules_11_modules_1_modules_layers_modules_3_parameters_weight_
        l_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_buffers_running_mean_ = L_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_buffers_running_mean_
        l_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_buffers_running_var_ = L_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_buffers_running_var_
        l_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_parameters_weight_ = L_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_parameters_weight_
        l_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_parameters_bias_ = L_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_parameters_bias_
        l_self_modules_layers_modules_11_modules_1_modules_layers_modules_6_parameters_weight_ = L_self_modules_layers_modules_11_modules_1_modules_layers_modules_6_parameters_weight_
        l_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_buffers_running_mean_ = L_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_buffers_running_mean_
        l_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_buffers_running_var_ = L_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_buffers_running_var_
        l_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_parameters_weight_ = L_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_parameters_weight_
        l_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_parameters_bias_ = L_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_parameters_bias_
        l_self_modules_layers_modules_12_modules_0_modules_layers_modules_0_parameters_weight_ = L_self_modules_layers_modules_12_modules_0_modules_layers_modules_0_parameters_weight_
        l_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_buffers_running_mean_ = L_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_buffers_running_mean_
        l_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_buffers_running_var_ = L_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_buffers_running_var_
        l_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_parameters_weight_ = L_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_parameters_weight_
        l_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_parameters_bias_ = L_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_parameters_bias_
        l_self_modules_layers_modules_12_modules_0_modules_layers_modules_3_parameters_weight_ = L_self_modules_layers_modules_12_modules_0_modules_layers_modules_3_parameters_weight_
        l_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_buffers_running_mean_ = L_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_buffers_running_mean_
        l_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_buffers_running_var_ = L_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_buffers_running_var_
        l_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_parameters_weight_ = L_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_parameters_weight_
        l_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_parameters_bias_ = L_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_parameters_bias_
        l_self_modules_layers_modules_12_modules_0_modules_layers_modules_6_parameters_weight_ = L_self_modules_layers_modules_12_modules_0_modules_layers_modules_6_parameters_weight_
        l_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_buffers_running_mean_ = L_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_buffers_running_mean_
        l_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_buffers_running_var_ = L_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_buffers_running_var_
        l_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_parameters_weight_ = L_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_parameters_weight_
        l_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_parameters_bias_ = L_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_parameters_bias_
        l_self_modules_layers_modules_12_modules_1_modules_layers_modules_0_parameters_weight_ = L_self_modules_layers_modules_12_modules_1_modules_layers_modules_0_parameters_weight_
        l_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_buffers_running_mean_ = L_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_buffers_running_mean_
        l_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_buffers_running_var_ = L_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_buffers_running_var_
        l_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_parameters_weight_ = L_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_parameters_weight_
        l_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_parameters_bias_ = L_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_parameters_bias_
        l_self_modules_layers_modules_12_modules_1_modules_layers_modules_3_parameters_weight_ = L_self_modules_layers_modules_12_modules_1_modules_layers_modules_3_parameters_weight_
        l_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_buffers_running_mean_ = L_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_buffers_running_mean_
        l_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_buffers_running_var_ = L_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_buffers_running_var_
        l_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_parameters_weight_ = L_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_parameters_weight_
        l_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_parameters_bias_ = L_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_parameters_bias_
        l_self_modules_layers_modules_12_modules_1_modules_layers_modules_6_parameters_weight_ = L_self_modules_layers_modules_12_modules_1_modules_layers_modules_6_parameters_weight_
        l_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_buffers_running_mean_ = L_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_buffers_running_mean_
        l_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_buffers_running_var_ = L_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_buffers_running_var_
        l_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_parameters_weight_ = L_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_parameters_weight_
        l_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_parameters_bias_ = L_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_parameters_bias_
        l_self_modules_layers_modules_12_modules_2_modules_layers_modules_0_parameters_weight_ = L_self_modules_layers_modules_12_modules_2_modules_layers_modules_0_parameters_weight_
        l_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_buffers_running_mean_ = L_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_buffers_running_mean_
        l_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_buffers_running_var_ = L_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_buffers_running_var_
        l_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_parameters_weight_ = L_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_parameters_weight_
        l_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_parameters_bias_ = L_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_parameters_bias_
        l_self_modules_layers_modules_12_modules_2_modules_layers_modules_3_parameters_weight_ = L_self_modules_layers_modules_12_modules_2_modules_layers_modules_3_parameters_weight_
        l_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_buffers_running_mean_ = L_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_buffers_running_mean_
        l_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_buffers_running_var_ = L_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_buffers_running_var_
        l_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_parameters_weight_ = L_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_parameters_weight_
        l_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_parameters_bias_ = L_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_parameters_bias_
        l_self_modules_layers_modules_12_modules_2_modules_layers_modules_6_parameters_weight_ = L_self_modules_layers_modules_12_modules_2_modules_layers_modules_6_parameters_weight_
        l_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_buffers_running_mean_ = L_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_buffers_running_mean_
        l_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_buffers_running_var_ = L_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_buffers_running_var_
        l_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_parameters_weight_ = L_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_parameters_weight_
        l_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_parameters_bias_ = L_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_parameters_bias_
        l_self_modules_layers_modules_12_modules_3_modules_layers_modules_0_parameters_weight_ = L_self_modules_layers_modules_12_modules_3_modules_layers_modules_0_parameters_weight_
        l_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_buffers_running_mean_ = L_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_buffers_running_mean_
        l_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_buffers_running_var_ = L_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_buffers_running_var_
        l_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_parameters_weight_ = L_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_parameters_weight_
        l_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_parameters_bias_ = L_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_parameters_bias_
        l_self_modules_layers_modules_12_modules_3_modules_layers_modules_3_parameters_weight_ = L_self_modules_layers_modules_12_modules_3_modules_layers_modules_3_parameters_weight_
        l_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_buffers_running_mean_ = L_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_buffers_running_mean_
        l_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_buffers_running_var_ = L_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_buffers_running_var_
        l_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_parameters_weight_ = L_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_parameters_weight_
        l_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_parameters_bias_ = L_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_parameters_bias_
        l_self_modules_layers_modules_12_modules_3_modules_layers_modules_6_parameters_weight_ = L_self_modules_layers_modules_12_modules_3_modules_layers_modules_6_parameters_weight_
        l_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_buffers_running_mean_ = L_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_buffers_running_mean_
        l_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_buffers_running_var_ = L_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_buffers_running_var_
        l_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_parameters_weight_ = L_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_parameters_weight_
        l_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_parameters_bias_ = L_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_parameters_bias_
        l_self_modules_layers_modules_13_modules_0_modules_layers_modules_0_parameters_weight_ = L_self_modules_layers_modules_13_modules_0_modules_layers_modules_0_parameters_weight_
        l_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_buffers_running_mean_ = L_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_buffers_running_mean_
        l_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_buffers_running_var_ = L_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_buffers_running_var_
        l_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_parameters_weight_ = L_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_parameters_weight_
        l_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_parameters_bias_ = L_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_parameters_bias_
        l_self_modules_layers_modules_13_modules_0_modules_layers_modules_3_parameters_weight_ = L_self_modules_layers_modules_13_modules_0_modules_layers_modules_3_parameters_weight_
        l_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_buffers_running_mean_ = L_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_buffers_running_mean_
        l_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_buffers_running_var_ = L_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_buffers_running_var_
        l_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_parameters_weight_ = L_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_parameters_weight_
        l_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_parameters_bias_ = L_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_parameters_bias_
        l_self_modules_layers_modules_13_modules_0_modules_layers_modules_6_parameters_weight_ = L_self_modules_layers_modules_13_modules_0_modules_layers_modules_6_parameters_weight_
        l_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_buffers_running_mean_ = L_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_buffers_running_mean_
        l_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_buffers_running_var_ = L_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_buffers_running_var_
        l_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_parameters_weight_ = L_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_parameters_weight_
        l_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_parameters_bias_ = L_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_parameters_bias_
        l_self_modules_layers_modules_14_parameters_weight_ = (
            L_self_modules_layers_modules_14_parameters_weight_
        )
        l_self_modules_layers_modules_15_buffers_running_mean_ = (
            L_self_modules_layers_modules_15_buffers_running_mean_
        )
        l_self_modules_layers_modules_15_buffers_running_var_ = (
            L_self_modules_layers_modules_15_buffers_running_var_
        )
        l_self_modules_layers_modules_15_parameters_weight_ = (
            L_self_modules_layers_modules_15_parameters_weight_
        )
        l_self_modules_layers_modules_15_parameters_bias_ = (
            L_self_modules_layers_modules_15_parameters_bias_
        )
        l_self_modules_classifier_modules_1_parameters_weight_ = (
            L_self_modules_classifier_modules_1_parameters_weight_
        )
        l_self_modules_classifier_modules_1_parameters_bias_ = (
            L_self_modules_classifier_modules_1_parameters_bias_
        )
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_layers_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_layers_modules_0_parameters_weight_ = None
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_layers_modules_1_buffers_running_mean_,
            l_self_modules_layers_modules_1_buffers_running_var_,
            l_self_modules_layers_modules_1_parameters_weight_,
            l_self_modules_layers_modules_1_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_1 = (
            l_self_modules_layers_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layers_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layers_modules_1_parameters_weight_
        ) = l_self_modules_layers_modules_1_parameters_bias_ = None
        input_3 = torch.nn.functional.relu(input_2, inplace=True)
        input_2 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_layers_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        input_3 = l_self_modules_layers_modules_3_parameters_weight_ = None
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_layers_modules_4_buffers_running_mean_,
            l_self_modules_layers_modules_4_buffers_running_var_,
            l_self_modules_layers_modules_4_parameters_weight_,
            l_self_modules_layers_modules_4_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_4 = (
            l_self_modules_layers_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_layers_modules_4_buffers_running_var_
        ) = (
            l_self_modules_layers_modules_4_parameters_weight_
        ) = l_self_modules_layers_modules_4_parameters_bias_ = None
        input_6 = torch.nn.functional.relu(input_5, inplace=True)
        input_5 = None
        input_7 = torch.conv2d(
            input_6,
            l_self_modules_layers_modules_6_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_6 = l_self_modules_layers_modules_6_parameters_weight_ = None
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_layers_modules_7_buffers_running_mean_,
            l_self_modules_layers_modules_7_buffers_running_var_,
            l_self_modules_layers_modules_7_parameters_weight_,
            l_self_modules_layers_modules_7_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_7 = (
            l_self_modules_layers_modules_7_buffers_running_mean_
        ) = (
            l_self_modules_layers_modules_7_buffers_running_var_
        ) = (
            l_self_modules_layers_modules_7_parameters_weight_
        ) = l_self_modules_layers_modules_7_parameters_bias_ = None
        input_9 = torch.conv2d(
            input_8,
            l_self_modules_layers_modules_8_modules_0_modules_layers_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_8 = l_self_modules_layers_modules_8_modules_0_modules_layers_modules_0_parameters_weight_ = (None)
        input_10 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_buffers_running_mean_,
            l_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_buffers_running_var_,
            l_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_parameters_weight_,
            l_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_9 = l_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_buffers_running_mean_ = l_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_buffers_running_var_ = l_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_parameters_weight_ = l_self_modules_layers_modules_8_modules_0_modules_layers_modules_1_parameters_bias_ = (None)
        input_11 = torch.nn.functional.relu(input_10, inplace=True)
        input_10 = None
        input_12 = torch.conv2d(
            input_11,
            l_self_modules_layers_modules_8_modules_0_modules_layers_modules_3_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            48,
        )
        input_11 = l_self_modules_layers_modules_8_modules_0_modules_layers_modules_3_parameters_weight_ = (None)
        input_13 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_buffers_running_mean_,
            l_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_buffers_running_var_,
            l_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_parameters_weight_,
            l_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_12 = l_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_buffers_running_mean_ = l_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_buffers_running_var_ = l_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_parameters_weight_ = l_self_modules_layers_modules_8_modules_0_modules_layers_modules_4_parameters_bias_ = (None)
        input_14 = torch.nn.functional.relu(input_13, inplace=True)
        input_13 = None
        input_15 = torch.conv2d(
            input_14,
            l_self_modules_layers_modules_8_modules_0_modules_layers_modules_6_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_14 = l_self_modules_layers_modules_8_modules_0_modules_layers_modules_6_parameters_weight_ = (None)
        input_16 = torch.nn.functional.batch_norm(
            input_15,
            l_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_buffers_running_mean_,
            l_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_buffers_running_var_,
            l_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_parameters_weight_,
            l_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_15 = l_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_buffers_running_mean_ = l_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_buffers_running_var_ = l_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_parameters_weight_ = l_self_modules_layers_modules_8_modules_0_modules_layers_modules_7_parameters_bias_ = (None)
        input_17 = torch.conv2d(
            input_16,
            l_self_modules_layers_modules_8_modules_1_modules_layers_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_8_modules_1_modules_layers_modules_0_parameters_weight_ = (
            None
        )
        input_18 = torch.nn.functional.batch_norm(
            input_17,
            l_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_buffers_running_mean_,
            l_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_buffers_running_var_,
            l_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_parameters_weight_,
            l_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_17 = l_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_buffers_running_mean_ = l_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_buffers_running_var_ = l_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_parameters_weight_ = l_self_modules_layers_modules_8_modules_1_modules_layers_modules_1_parameters_bias_ = (None)
        input_19 = torch.nn.functional.relu(input_18, inplace=True)
        input_18 = None
        input_20 = torch.conv2d(
            input_19,
            l_self_modules_layers_modules_8_modules_1_modules_layers_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            72,
        )
        input_19 = l_self_modules_layers_modules_8_modules_1_modules_layers_modules_3_parameters_weight_ = (None)
        input_21 = torch.nn.functional.batch_norm(
            input_20,
            l_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_buffers_running_mean_,
            l_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_buffers_running_var_,
            l_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_parameters_weight_,
            l_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_20 = l_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_buffers_running_mean_ = l_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_buffers_running_var_ = l_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_parameters_weight_ = l_self_modules_layers_modules_8_modules_1_modules_layers_modules_4_parameters_bias_ = (None)
        input_22 = torch.nn.functional.relu(input_21, inplace=True)
        input_21 = None
        input_23 = torch.conv2d(
            input_22,
            l_self_modules_layers_modules_8_modules_1_modules_layers_modules_6_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_22 = l_self_modules_layers_modules_8_modules_1_modules_layers_modules_6_parameters_weight_ = (None)
        input_24 = torch.nn.functional.batch_norm(
            input_23,
            l_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_buffers_running_mean_,
            l_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_buffers_running_var_,
            l_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_parameters_weight_,
            l_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_23 = l_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_buffers_running_mean_ = l_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_buffers_running_var_ = l_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_parameters_weight_ = l_self_modules_layers_modules_8_modules_1_modules_layers_modules_7_parameters_bias_ = (None)
        input_25 = input_24 + input_16
        input_24 = input_16 = None
        input_26 = torch.conv2d(
            input_25,
            l_self_modules_layers_modules_8_modules_2_modules_layers_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_8_modules_2_modules_layers_modules_0_parameters_weight_ = (
            None
        )
        input_27 = torch.nn.functional.batch_norm(
            input_26,
            l_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_buffers_running_mean_,
            l_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_buffers_running_var_,
            l_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_parameters_weight_,
            l_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_26 = l_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_buffers_running_mean_ = l_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_buffers_running_var_ = l_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_parameters_weight_ = l_self_modules_layers_modules_8_modules_2_modules_layers_modules_1_parameters_bias_ = (None)
        input_28 = torch.nn.functional.relu(input_27, inplace=True)
        input_27 = None
        input_29 = torch.conv2d(
            input_28,
            l_self_modules_layers_modules_8_modules_2_modules_layers_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            72,
        )
        input_28 = l_self_modules_layers_modules_8_modules_2_modules_layers_modules_3_parameters_weight_ = (None)
        input_30 = torch.nn.functional.batch_norm(
            input_29,
            l_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_buffers_running_mean_,
            l_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_buffers_running_var_,
            l_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_parameters_weight_,
            l_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_29 = l_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_buffers_running_mean_ = l_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_buffers_running_var_ = l_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_parameters_weight_ = l_self_modules_layers_modules_8_modules_2_modules_layers_modules_4_parameters_bias_ = (None)
        input_31 = torch.nn.functional.relu(input_30, inplace=True)
        input_30 = None
        input_32 = torch.conv2d(
            input_31,
            l_self_modules_layers_modules_8_modules_2_modules_layers_modules_6_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_31 = l_self_modules_layers_modules_8_modules_2_modules_layers_modules_6_parameters_weight_ = (None)
        input_33 = torch.nn.functional.batch_norm(
            input_32,
            l_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_buffers_running_mean_,
            l_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_buffers_running_var_,
            l_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_parameters_weight_,
            l_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_32 = l_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_buffers_running_mean_ = l_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_buffers_running_var_ = l_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_parameters_weight_ = l_self_modules_layers_modules_8_modules_2_modules_layers_modules_7_parameters_bias_ = (None)
        input_34 = input_33 + input_25
        input_33 = input_25 = None
        input_35 = torch.conv2d(
            input_34,
            l_self_modules_layers_modules_9_modules_0_modules_layers_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_34 = l_self_modules_layers_modules_9_modules_0_modules_layers_modules_0_parameters_weight_ = (None)
        input_36 = torch.nn.functional.batch_norm(
            input_35,
            l_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_buffers_running_mean_,
            l_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_buffers_running_var_,
            l_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_parameters_weight_,
            l_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_35 = l_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_buffers_running_mean_ = l_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_buffers_running_var_ = l_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_parameters_weight_ = l_self_modules_layers_modules_9_modules_0_modules_layers_modules_1_parameters_bias_ = (None)
        input_37 = torch.nn.functional.relu(input_36, inplace=True)
        input_36 = None
        input_38 = torch.conv2d(
            input_37,
            l_self_modules_layers_modules_9_modules_0_modules_layers_modules_3_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            72,
        )
        input_37 = l_self_modules_layers_modules_9_modules_0_modules_layers_modules_3_parameters_weight_ = (None)
        input_39 = torch.nn.functional.batch_norm(
            input_38,
            l_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_buffers_running_mean_,
            l_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_buffers_running_var_,
            l_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_parameters_weight_,
            l_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_38 = l_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_buffers_running_mean_ = l_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_buffers_running_var_ = l_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_parameters_weight_ = l_self_modules_layers_modules_9_modules_0_modules_layers_modules_4_parameters_bias_ = (None)
        input_40 = torch.nn.functional.relu(input_39, inplace=True)
        input_39 = None
        input_41 = torch.conv2d(
            input_40,
            l_self_modules_layers_modules_9_modules_0_modules_layers_modules_6_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_40 = l_self_modules_layers_modules_9_modules_0_modules_layers_modules_6_parameters_weight_ = (None)
        input_42 = torch.nn.functional.batch_norm(
            input_41,
            l_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_buffers_running_mean_,
            l_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_buffers_running_var_,
            l_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_parameters_weight_,
            l_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_41 = l_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_buffers_running_mean_ = l_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_buffers_running_var_ = l_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_parameters_weight_ = l_self_modules_layers_modules_9_modules_0_modules_layers_modules_7_parameters_bias_ = (None)
        input_43 = torch.conv2d(
            input_42,
            l_self_modules_layers_modules_9_modules_1_modules_layers_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_9_modules_1_modules_layers_modules_0_parameters_weight_ = (
            None
        )
        input_44 = torch.nn.functional.batch_norm(
            input_43,
            l_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_buffers_running_mean_,
            l_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_buffers_running_var_,
            l_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_parameters_weight_,
            l_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_43 = l_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_buffers_running_mean_ = l_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_buffers_running_var_ = l_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_parameters_weight_ = l_self_modules_layers_modules_9_modules_1_modules_layers_modules_1_parameters_bias_ = (None)
        input_45 = torch.nn.functional.relu(input_44, inplace=True)
        input_44 = None
        input_46 = torch.conv2d(
            input_45,
            l_self_modules_layers_modules_9_modules_1_modules_layers_modules_3_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        input_45 = l_self_modules_layers_modules_9_modules_1_modules_layers_modules_3_parameters_weight_ = (None)
        input_47 = torch.nn.functional.batch_norm(
            input_46,
            l_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_buffers_running_mean_,
            l_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_buffers_running_var_,
            l_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_parameters_weight_,
            l_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_46 = l_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_buffers_running_mean_ = l_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_buffers_running_var_ = l_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_parameters_weight_ = l_self_modules_layers_modules_9_modules_1_modules_layers_modules_4_parameters_bias_ = (None)
        input_48 = torch.nn.functional.relu(input_47, inplace=True)
        input_47 = None
        input_49 = torch.conv2d(
            input_48,
            l_self_modules_layers_modules_9_modules_1_modules_layers_modules_6_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_48 = l_self_modules_layers_modules_9_modules_1_modules_layers_modules_6_parameters_weight_ = (None)
        input_50 = torch.nn.functional.batch_norm(
            input_49,
            l_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_buffers_running_mean_,
            l_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_buffers_running_var_,
            l_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_parameters_weight_,
            l_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_49 = l_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_buffers_running_mean_ = l_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_buffers_running_var_ = l_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_parameters_weight_ = l_self_modules_layers_modules_9_modules_1_modules_layers_modules_7_parameters_bias_ = (None)
        input_51 = input_50 + input_42
        input_50 = input_42 = None
        input_52 = torch.conv2d(
            input_51,
            l_self_modules_layers_modules_9_modules_2_modules_layers_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_9_modules_2_modules_layers_modules_0_parameters_weight_ = (
            None
        )
        input_53 = torch.nn.functional.batch_norm(
            input_52,
            l_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_buffers_running_mean_,
            l_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_buffers_running_var_,
            l_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_parameters_weight_,
            l_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_52 = l_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_buffers_running_mean_ = l_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_buffers_running_var_ = l_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_parameters_weight_ = l_self_modules_layers_modules_9_modules_2_modules_layers_modules_1_parameters_bias_ = (None)
        input_54 = torch.nn.functional.relu(input_53, inplace=True)
        input_53 = None
        input_55 = torch.conv2d(
            input_54,
            l_self_modules_layers_modules_9_modules_2_modules_layers_modules_3_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        input_54 = l_self_modules_layers_modules_9_modules_2_modules_layers_modules_3_parameters_weight_ = (None)
        input_56 = torch.nn.functional.batch_norm(
            input_55,
            l_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_buffers_running_mean_,
            l_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_buffers_running_var_,
            l_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_parameters_weight_,
            l_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_55 = l_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_buffers_running_mean_ = l_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_buffers_running_var_ = l_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_parameters_weight_ = l_self_modules_layers_modules_9_modules_2_modules_layers_modules_4_parameters_bias_ = (None)
        input_57 = torch.nn.functional.relu(input_56, inplace=True)
        input_56 = None
        input_58 = torch.conv2d(
            input_57,
            l_self_modules_layers_modules_9_modules_2_modules_layers_modules_6_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_57 = l_self_modules_layers_modules_9_modules_2_modules_layers_modules_6_parameters_weight_ = (None)
        input_59 = torch.nn.functional.batch_norm(
            input_58,
            l_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_buffers_running_mean_,
            l_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_buffers_running_var_,
            l_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_parameters_weight_,
            l_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_58 = l_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_buffers_running_mean_ = l_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_buffers_running_var_ = l_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_parameters_weight_ = l_self_modules_layers_modules_9_modules_2_modules_layers_modules_7_parameters_bias_ = (None)
        input_60 = input_59 + input_51
        input_59 = input_51 = None
        input_61 = torch.conv2d(
            input_60,
            l_self_modules_layers_modules_10_modules_0_modules_layers_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_60 = l_self_modules_layers_modules_10_modules_0_modules_layers_modules_0_parameters_weight_ = (None)
        input_62 = torch.nn.functional.batch_norm(
            input_61,
            l_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_buffers_running_mean_,
            l_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_buffers_running_var_,
            l_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_parameters_weight_,
            l_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_61 = l_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_buffers_running_mean_ = l_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_buffers_running_var_ = l_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_parameters_weight_ = l_self_modules_layers_modules_10_modules_0_modules_layers_modules_1_parameters_bias_ = (None)
        input_63 = torch.nn.functional.relu(input_62, inplace=True)
        input_62 = None
        input_64 = torch.conv2d(
            input_63,
            l_self_modules_layers_modules_10_modules_0_modules_layers_modules_3_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            240,
        )
        input_63 = l_self_modules_layers_modules_10_modules_0_modules_layers_modules_3_parameters_weight_ = (None)
        input_65 = torch.nn.functional.batch_norm(
            input_64,
            l_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_buffers_running_mean_,
            l_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_buffers_running_var_,
            l_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_parameters_weight_,
            l_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_64 = l_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_buffers_running_mean_ = l_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_buffers_running_var_ = l_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_parameters_weight_ = l_self_modules_layers_modules_10_modules_0_modules_layers_modules_4_parameters_bias_ = (None)
        input_66 = torch.nn.functional.relu(input_65, inplace=True)
        input_65 = None
        input_67 = torch.conv2d(
            input_66,
            l_self_modules_layers_modules_10_modules_0_modules_layers_modules_6_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_66 = l_self_modules_layers_modules_10_modules_0_modules_layers_modules_6_parameters_weight_ = (None)
        input_68 = torch.nn.functional.batch_norm(
            input_67,
            l_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_buffers_running_mean_,
            l_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_buffers_running_var_,
            l_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_parameters_weight_,
            l_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_67 = l_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_buffers_running_mean_ = l_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_buffers_running_var_ = l_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_parameters_weight_ = l_self_modules_layers_modules_10_modules_0_modules_layers_modules_7_parameters_bias_ = (None)
        input_69 = torch.conv2d(
            input_68,
            l_self_modules_layers_modules_10_modules_1_modules_layers_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_10_modules_1_modules_layers_modules_0_parameters_weight_ = (
            None
        )
        input_70 = torch.nn.functional.batch_norm(
            input_69,
            l_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_buffers_running_mean_,
            l_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_buffers_running_var_,
            l_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_parameters_weight_,
            l_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_69 = l_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_buffers_running_mean_ = l_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_buffers_running_var_ = l_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_parameters_weight_ = l_self_modules_layers_modules_10_modules_1_modules_layers_modules_1_parameters_bias_ = (None)
        input_71 = torch.nn.functional.relu(input_70, inplace=True)
        input_70 = None
        input_72 = torch.conv2d(
            input_71,
            l_self_modules_layers_modules_10_modules_1_modules_layers_modules_3_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            480,
        )
        input_71 = l_self_modules_layers_modules_10_modules_1_modules_layers_modules_3_parameters_weight_ = (None)
        input_73 = torch.nn.functional.batch_norm(
            input_72,
            l_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_buffers_running_mean_,
            l_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_buffers_running_var_,
            l_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_parameters_weight_,
            l_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_72 = l_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_buffers_running_mean_ = l_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_buffers_running_var_ = l_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_parameters_weight_ = l_self_modules_layers_modules_10_modules_1_modules_layers_modules_4_parameters_bias_ = (None)
        input_74 = torch.nn.functional.relu(input_73, inplace=True)
        input_73 = None
        input_75 = torch.conv2d(
            input_74,
            l_self_modules_layers_modules_10_modules_1_modules_layers_modules_6_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_74 = l_self_modules_layers_modules_10_modules_1_modules_layers_modules_6_parameters_weight_ = (None)
        input_76 = torch.nn.functional.batch_norm(
            input_75,
            l_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_buffers_running_mean_,
            l_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_buffers_running_var_,
            l_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_parameters_weight_,
            l_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_75 = l_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_buffers_running_mean_ = l_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_buffers_running_var_ = l_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_parameters_weight_ = l_self_modules_layers_modules_10_modules_1_modules_layers_modules_7_parameters_bias_ = (None)
        input_77 = input_76 + input_68
        input_76 = input_68 = None
        input_78 = torch.conv2d(
            input_77,
            l_self_modules_layers_modules_10_modules_2_modules_layers_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_10_modules_2_modules_layers_modules_0_parameters_weight_ = (
            None
        )
        input_79 = torch.nn.functional.batch_norm(
            input_78,
            l_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_buffers_running_mean_,
            l_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_buffers_running_var_,
            l_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_parameters_weight_,
            l_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_78 = l_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_buffers_running_mean_ = l_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_buffers_running_var_ = l_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_parameters_weight_ = l_self_modules_layers_modules_10_modules_2_modules_layers_modules_1_parameters_bias_ = (None)
        input_80 = torch.nn.functional.relu(input_79, inplace=True)
        input_79 = None
        input_81 = torch.conv2d(
            input_80,
            l_self_modules_layers_modules_10_modules_2_modules_layers_modules_3_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            480,
        )
        input_80 = l_self_modules_layers_modules_10_modules_2_modules_layers_modules_3_parameters_weight_ = (None)
        input_82 = torch.nn.functional.batch_norm(
            input_81,
            l_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_buffers_running_mean_,
            l_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_buffers_running_var_,
            l_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_parameters_weight_,
            l_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_81 = l_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_buffers_running_mean_ = l_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_buffers_running_var_ = l_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_parameters_weight_ = l_self_modules_layers_modules_10_modules_2_modules_layers_modules_4_parameters_bias_ = (None)
        input_83 = torch.nn.functional.relu(input_82, inplace=True)
        input_82 = None
        input_84 = torch.conv2d(
            input_83,
            l_self_modules_layers_modules_10_modules_2_modules_layers_modules_6_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_83 = l_self_modules_layers_modules_10_modules_2_modules_layers_modules_6_parameters_weight_ = (None)
        input_85 = torch.nn.functional.batch_norm(
            input_84,
            l_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_buffers_running_mean_,
            l_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_buffers_running_var_,
            l_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_parameters_weight_,
            l_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_84 = l_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_buffers_running_mean_ = l_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_buffers_running_var_ = l_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_parameters_weight_ = l_self_modules_layers_modules_10_modules_2_modules_layers_modules_7_parameters_bias_ = (None)
        input_86 = input_85 + input_77
        input_85 = input_77 = None
        input_87 = torch.conv2d(
            input_86,
            l_self_modules_layers_modules_11_modules_0_modules_layers_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_86 = l_self_modules_layers_modules_11_modules_0_modules_layers_modules_0_parameters_weight_ = (None)
        input_88 = torch.nn.functional.batch_norm(
            input_87,
            l_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_buffers_running_mean_,
            l_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_buffers_running_var_,
            l_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_parameters_weight_,
            l_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_87 = l_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_buffers_running_mean_ = l_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_buffers_running_var_ = l_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_parameters_weight_ = l_self_modules_layers_modules_11_modules_0_modules_layers_modules_1_parameters_bias_ = (None)
        input_89 = torch.nn.functional.relu(input_88, inplace=True)
        input_88 = None
        input_90 = torch.conv2d(
            input_89,
            l_self_modules_layers_modules_11_modules_0_modules_layers_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            480,
        )
        input_89 = l_self_modules_layers_modules_11_modules_0_modules_layers_modules_3_parameters_weight_ = (None)
        input_91 = torch.nn.functional.batch_norm(
            input_90,
            l_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_buffers_running_mean_,
            l_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_buffers_running_var_,
            l_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_parameters_weight_,
            l_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_90 = l_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_buffers_running_mean_ = l_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_buffers_running_var_ = l_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_parameters_weight_ = l_self_modules_layers_modules_11_modules_0_modules_layers_modules_4_parameters_bias_ = (None)
        input_92 = torch.nn.functional.relu(input_91, inplace=True)
        input_91 = None
        input_93 = torch.conv2d(
            input_92,
            l_self_modules_layers_modules_11_modules_0_modules_layers_modules_6_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_92 = l_self_modules_layers_modules_11_modules_0_modules_layers_modules_6_parameters_weight_ = (None)
        input_94 = torch.nn.functional.batch_norm(
            input_93,
            l_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_buffers_running_mean_,
            l_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_buffers_running_var_,
            l_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_parameters_weight_,
            l_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_93 = l_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_buffers_running_mean_ = l_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_buffers_running_var_ = l_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_parameters_weight_ = l_self_modules_layers_modules_11_modules_0_modules_layers_modules_7_parameters_bias_ = (None)
        input_95 = torch.conv2d(
            input_94,
            l_self_modules_layers_modules_11_modules_1_modules_layers_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_11_modules_1_modules_layers_modules_0_parameters_weight_ = (
            None
        )
        input_96 = torch.nn.functional.batch_norm(
            input_95,
            l_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_buffers_running_mean_,
            l_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_buffers_running_var_,
            l_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_parameters_weight_,
            l_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_95 = l_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_buffers_running_mean_ = l_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_buffers_running_var_ = l_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_parameters_weight_ = l_self_modules_layers_modules_11_modules_1_modules_layers_modules_1_parameters_bias_ = (None)
        input_97 = torch.nn.functional.relu(input_96, inplace=True)
        input_96 = None
        input_98 = torch.conv2d(
            input_97,
            l_self_modules_layers_modules_11_modules_1_modules_layers_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            576,
        )
        input_97 = l_self_modules_layers_modules_11_modules_1_modules_layers_modules_3_parameters_weight_ = (None)
        input_99 = torch.nn.functional.batch_norm(
            input_98,
            l_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_buffers_running_mean_,
            l_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_buffers_running_var_,
            l_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_parameters_weight_,
            l_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_98 = l_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_buffers_running_mean_ = l_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_buffers_running_var_ = l_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_parameters_weight_ = l_self_modules_layers_modules_11_modules_1_modules_layers_modules_4_parameters_bias_ = (None)
        input_100 = torch.nn.functional.relu(input_99, inplace=True)
        input_99 = None
        input_101 = torch.conv2d(
            input_100,
            l_self_modules_layers_modules_11_modules_1_modules_layers_modules_6_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_100 = l_self_modules_layers_modules_11_modules_1_modules_layers_modules_6_parameters_weight_ = (None)
        input_102 = torch.nn.functional.batch_norm(
            input_101,
            l_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_buffers_running_mean_,
            l_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_buffers_running_var_,
            l_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_parameters_weight_,
            l_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_101 = l_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_buffers_running_mean_ = l_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_buffers_running_var_ = l_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_parameters_weight_ = l_self_modules_layers_modules_11_modules_1_modules_layers_modules_7_parameters_bias_ = (None)
        input_103 = input_102 + input_94
        input_102 = input_94 = None
        input_104 = torch.conv2d(
            input_103,
            l_self_modules_layers_modules_12_modules_0_modules_layers_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_103 = l_self_modules_layers_modules_12_modules_0_modules_layers_modules_0_parameters_weight_ = (None)
        input_105 = torch.nn.functional.batch_norm(
            input_104,
            l_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_buffers_running_mean_,
            l_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_buffers_running_var_,
            l_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_parameters_weight_,
            l_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_104 = l_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_buffers_running_mean_ = l_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_buffers_running_var_ = l_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_parameters_weight_ = l_self_modules_layers_modules_12_modules_0_modules_layers_modules_1_parameters_bias_ = (None)
        input_106 = torch.nn.functional.relu(input_105, inplace=True)
        input_105 = None
        input_107 = torch.conv2d(
            input_106,
            l_self_modules_layers_modules_12_modules_0_modules_layers_modules_3_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            576,
        )
        input_106 = l_self_modules_layers_modules_12_modules_0_modules_layers_modules_3_parameters_weight_ = (None)
        input_108 = torch.nn.functional.batch_norm(
            input_107,
            l_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_buffers_running_mean_,
            l_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_buffers_running_var_,
            l_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_parameters_weight_,
            l_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_107 = l_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_buffers_running_mean_ = l_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_buffers_running_var_ = l_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_parameters_weight_ = l_self_modules_layers_modules_12_modules_0_modules_layers_modules_4_parameters_bias_ = (None)
        input_109 = torch.nn.functional.relu(input_108, inplace=True)
        input_108 = None
        input_110 = torch.conv2d(
            input_109,
            l_self_modules_layers_modules_12_modules_0_modules_layers_modules_6_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_109 = l_self_modules_layers_modules_12_modules_0_modules_layers_modules_6_parameters_weight_ = (None)
        input_111 = torch.nn.functional.batch_norm(
            input_110,
            l_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_buffers_running_mean_,
            l_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_buffers_running_var_,
            l_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_parameters_weight_,
            l_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_110 = l_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_buffers_running_mean_ = l_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_buffers_running_var_ = l_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_parameters_weight_ = l_self_modules_layers_modules_12_modules_0_modules_layers_modules_7_parameters_bias_ = (None)
        input_112 = torch.conv2d(
            input_111,
            l_self_modules_layers_modules_12_modules_1_modules_layers_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_12_modules_1_modules_layers_modules_0_parameters_weight_ = (
            None
        )
        input_113 = torch.nn.functional.batch_norm(
            input_112,
            l_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_buffers_running_mean_,
            l_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_buffers_running_var_,
            l_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_parameters_weight_,
            l_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_112 = l_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_buffers_running_mean_ = l_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_buffers_running_var_ = l_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_parameters_weight_ = l_self_modules_layers_modules_12_modules_1_modules_layers_modules_1_parameters_bias_ = (None)
        input_114 = torch.nn.functional.relu(input_113, inplace=True)
        input_113 = None
        input_115 = torch.conv2d(
            input_114,
            l_self_modules_layers_modules_12_modules_1_modules_layers_modules_3_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1152,
        )
        input_114 = l_self_modules_layers_modules_12_modules_1_modules_layers_modules_3_parameters_weight_ = (None)
        input_116 = torch.nn.functional.batch_norm(
            input_115,
            l_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_buffers_running_mean_,
            l_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_buffers_running_var_,
            l_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_parameters_weight_,
            l_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_115 = l_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_buffers_running_mean_ = l_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_buffers_running_var_ = l_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_parameters_weight_ = l_self_modules_layers_modules_12_modules_1_modules_layers_modules_4_parameters_bias_ = (None)
        input_117 = torch.nn.functional.relu(input_116, inplace=True)
        input_116 = None
        input_118 = torch.conv2d(
            input_117,
            l_self_modules_layers_modules_12_modules_1_modules_layers_modules_6_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_117 = l_self_modules_layers_modules_12_modules_1_modules_layers_modules_6_parameters_weight_ = (None)
        input_119 = torch.nn.functional.batch_norm(
            input_118,
            l_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_buffers_running_mean_,
            l_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_buffers_running_var_,
            l_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_parameters_weight_,
            l_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_118 = l_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_buffers_running_mean_ = l_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_buffers_running_var_ = l_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_parameters_weight_ = l_self_modules_layers_modules_12_modules_1_modules_layers_modules_7_parameters_bias_ = (None)
        input_120 = input_119 + input_111
        input_119 = input_111 = None
        input_121 = torch.conv2d(
            input_120,
            l_self_modules_layers_modules_12_modules_2_modules_layers_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_12_modules_2_modules_layers_modules_0_parameters_weight_ = (
            None
        )
        input_122 = torch.nn.functional.batch_norm(
            input_121,
            l_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_buffers_running_mean_,
            l_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_buffers_running_var_,
            l_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_parameters_weight_,
            l_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_121 = l_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_buffers_running_mean_ = l_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_buffers_running_var_ = l_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_parameters_weight_ = l_self_modules_layers_modules_12_modules_2_modules_layers_modules_1_parameters_bias_ = (None)
        input_123 = torch.nn.functional.relu(input_122, inplace=True)
        input_122 = None
        input_124 = torch.conv2d(
            input_123,
            l_self_modules_layers_modules_12_modules_2_modules_layers_modules_3_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1152,
        )
        input_123 = l_self_modules_layers_modules_12_modules_2_modules_layers_modules_3_parameters_weight_ = (None)
        input_125 = torch.nn.functional.batch_norm(
            input_124,
            l_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_buffers_running_mean_,
            l_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_buffers_running_var_,
            l_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_parameters_weight_,
            l_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_124 = l_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_buffers_running_mean_ = l_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_buffers_running_var_ = l_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_parameters_weight_ = l_self_modules_layers_modules_12_modules_2_modules_layers_modules_4_parameters_bias_ = (None)
        input_126 = torch.nn.functional.relu(input_125, inplace=True)
        input_125 = None
        input_127 = torch.conv2d(
            input_126,
            l_self_modules_layers_modules_12_modules_2_modules_layers_modules_6_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_126 = l_self_modules_layers_modules_12_modules_2_modules_layers_modules_6_parameters_weight_ = (None)
        input_128 = torch.nn.functional.batch_norm(
            input_127,
            l_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_buffers_running_mean_,
            l_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_buffers_running_var_,
            l_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_parameters_weight_,
            l_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_127 = l_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_buffers_running_mean_ = l_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_buffers_running_var_ = l_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_parameters_weight_ = l_self_modules_layers_modules_12_modules_2_modules_layers_modules_7_parameters_bias_ = (None)
        input_129 = input_128 + input_120
        input_128 = input_120 = None
        input_130 = torch.conv2d(
            input_129,
            l_self_modules_layers_modules_12_modules_3_modules_layers_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layers_modules_12_modules_3_modules_layers_modules_0_parameters_weight_ = (
            None
        )
        input_131 = torch.nn.functional.batch_norm(
            input_130,
            l_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_buffers_running_mean_,
            l_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_buffers_running_var_,
            l_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_parameters_weight_,
            l_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_130 = l_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_buffers_running_mean_ = l_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_buffers_running_var_ = l_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_parameters_weight_ = l_self_modules_layers_modules_12_modules_3_modules_layers_modules_1_parameters_bias_ = (None)
        input_132 = torch.nn.functional.relu(input_131, inplace=True)
        input_131 = None
        input_133 = torch.conv2d(
            input_132,
            l_self_modules_layers_modules_12_modules_3_modules_layers_modules_3_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1152,
        )
        input_132 = l_self_modules_layers_modules_12_modules_3_modules_layers_modules_3_parameters_weight_ = (None)
        input_134 = torch.nn.functional.batch_norm(
            input_133,
            l_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_buffers_running_mean_,
            l_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_buffers_running_var_,
            l_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_parameters_weight_,
            l_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_133 = l_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_buffers_running_mean_ = l_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_buffers_running_var_ = l_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_parameters_weight_ = l_self_modules_layers_modules_12_modules_3_modules_layers_modules_4_parameters_bias_ = (None)
        input_135 = torch.nn.functional.relu(input_134, inplace=True)
        input_134 = None
        input_136 = torch.conv2d(
            input_135,
            l_self_modules_layers_modules_12_modules_3_modules_layers_modules_6_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_135 = l_self_modules_layers_modules_12_modules_3_modules_layers_modules_6_parameters_weight_ = (None)
        input_137 = torch.nn.functional.batch_norm(
            input_136,
            l_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_buffers_running_mean_,
            l_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_buffers_running_var_,
            l_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_parameters_weight_,
            l_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_136 = l_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_buffers_running_mean_ = l_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_buffers_running_var_ = l_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_parameters_weight_ = l_self_modules_layers_modules_12_modules_3_modules_layers_modules_7_parameters_bias_ = (None)
        input_138 = input_137 + input_129
        input_137 = input_129 = None
        input_139 = torch.conv2d(
            input_138,
            l_self_modules_layers_modules_13_modules_0_modules_layers_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_138 = l_self_modules_layers_modules_13_modules_0_modules_layers_modules_0_parameters_weight_ = (None)
        input_140 = torch.nn.functional.batch_norm(
            input_139,
            l_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_buffers_running_mean_,
            l_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_buffers_running_var_,
            l_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_parameters_weight_,
            l_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_139 = l_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_buffers_running_mean_ = l_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_buffers_running_var_ = l_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_parameters_weight_ = l_self_modules_layers_modules_13_modules_0_modules_layers_modules_1_parameters_bias_ = (None)
        input_141 = torch.nn.functional.relu(input_140, inplace=True)
        input_140 = None
        input_142 = torch.conv2d(
            input_141,
            l_self_modules_layers_modules_13_modules_0_modules_layers_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1152,
        )
        input_141 = l_self_modules_layers_modules_13_modules_0_modules_layers_modules_3_parameters_weight_ = (None)
        input_143 = torch.nn.functional.batch_norm(
            input_142,
            l_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_buffers_running_mean_,
            l_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_buffers_running_var_,
            l_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_parameters_weight_,
            l_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_142 = l_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_buffers_running_mean_ = l_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_buffers_running_var_ = l_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_parameters_weight_ = l_self_modules_layers_modules_13_modules_0_modules_layers_modules_4_parameters_bias_ = (None)
        input_144 = torch.nn.functional.relu(input_143, inplace=True)
        input_143 = None
        input_145 = torch.conv2d(
            input_144,
            l_self_modules_layers_modules_13_modules_0_modules_layers_modules_6_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_144 = l_self_modules_layers_modules_13_modules_0_modules_layers_modules_6_parameters_weight_ = (None)
        input_146 = torch.nn.functional.batch_norm(
            input_145,
            l_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_buffers_running_mean_,
            l_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_buffers_running_var_,
            l_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_parameters_weight_,
            l_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_145 = l_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_buffers_running_mean_ = l_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_buffers_running_var_ = l_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_parameters_weight_ = l_self_modules_layers_modules_13_modules_0_modules_layers_modules_7_parameters_bias_ = (None)
        input_147 = torch.conv2d(
            input_146,
            l_self_modules_layers_modules_14_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_146 = l_self_modules_layers_modules_14_parameters_weight_ = None
        input_148 = torch.nn.functional.batch_norm(
            input_147,
            l_self_modules_layers_modules_15_buffers_running_mean_,
            l_self_modules_layers_modules_15_buffers_running_var_,
            l_self_modules_layers_modules_15_parameters_weight_,
            l_self_modules_layers_modules_15_parameters_bias_,
            False,
            0.00029999999999996696,
            1e-05,
        )
        input_147 = (
            l_self_modules_layers_modules_15_buffers_running_mean_
        ) = (
            l_self_modules_layers_modules_15_buffers_running_var_
        ) = (
            l_self_modules_layers_modules_15_parameters_weight_
        ) = l_self_modules_layers_modules_15_parameters_bias_ = None
        input_149 = torch.nn.functional.relu(input_148, inplace=True)
        input_148 = None
        x = input_149.mean([2, 3])
        input_149 = None
        input_150 = torch.nn.functional.dropout(x, 0.2, False, True)
        x = None
        input_151 = torch._C._nn.linear(
            input_150,
            l_self_modules_classifier_modules_1_parameters_weight_,
            l_self_modules_classifier_modules_1_parameters_bias_,
        )
        input_150 = (
            l_self_modules_classifier_modules_1_parameters_weight_
        ) = l_self_modules_classifier_modules_1_parameters_bias_ = None
        return (input_151,)
