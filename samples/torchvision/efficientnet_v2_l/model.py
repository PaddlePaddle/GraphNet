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
        L_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_8_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_8_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_8_modules_1_buffers_running_mean_ = (
            L_self_modules_features_modules_8_modules_1_buffers_running_mean_
        )
        l_self_modules_features_modules_8_modules_1_buffers_running_var_ = (
            L_self_modules_features_modules_8_modules_1_buffers_running_var_
        )
        l_self_modules_features_modules_8_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_8_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_8_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_8_modules_1_parameters_bias_
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
            0.001,
        )
        input_1 = (
            l_self_modules_features_modules_0_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_0_modules_1_buffers_running_var_
        ) = (
            l_self_modules_features_modules_0_modules_1_parameters_weight_
        ) = l_self_modules_features_modules_0_modules_1_parameters_bias_ = None
        input_3 = torch.nn.functional.silu(input_2, inplace=True)
        input_2 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_4 = l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_6 = torch.nn.functional.silu(input_5, inplace=True)
        input_5 = None
        _log_api_usage_once = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once = None
        input_6 += input_3
        result = input_6
        input_6 = input_3 = None
        input_7 = torch.conv2d(
            result,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_7 = l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_9 = torch.nn.functional.silu(input_8, inplace=True)
        input_8 = None
        _log_api_usage_once_1 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_1 = None
        input_9 += result
        result_1 = input_9
        input_9 = result = None
        input_10 = torch.conv2d(
            result_1,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_11 = torch.nn.functional.batch_norm(
            input_10,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_10 = l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_12 = torch.nn.functional.silu(input_11, inplace=True)
        input_11 = None
        _log_api_usage_once_2 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_2 = None
        input_12 += result_1
        result_2 = input_12
        input_12 = result_1 = None
        input_13 = torch.conv2d(
            result_2,
            l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_14 = torch.nn.functional.batch_norm(
            input_13,
            l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_13 = l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_15 = torch.nn.functional.silu(input_14, inplace=True)
        input_14 = None
        _log_api_usage_once_3 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_3 = None
        input_15 += result_2
        result_3 = input_15
        input_15 = result_2 = None
        input_16 = torch.conv2d(
            result_3,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        result_3 = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_17 = torch.nn.functional.batch_norm(
            input_16,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_16 = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_18 = torch.nn.functional.silu(input_17, inplace=True)
        input_17 = None
        input_19 = torch.conv2d(
            input_18,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_18 = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_20 = torch.nn.functional.batch_norm(
            input_19,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_19 = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_21 = torch.conv2d(
            input_20,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_22 = torch.nn.functional.batch_norm(
            input_21,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_21 = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_23 = torch.nn.functional.silu(input_22, inplace=True)
        input_22 = None
        input_24 = torch.conv2d(
            input_23,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_23 = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_25 = torch.nn.functional.batch_norm(
            input_24,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_24 = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_4 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_4 = None
        input_25 += input_20
        result_4 = input_25
        input_25 = input_20 = None
        input_26 = torch.conv2d(
            result_4,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_27 = torch.nn.functional.batch_norm(
            input_26,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_26 = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_28 = torch.nn.functional.silu(input_27, inplace=True)
        input_27 = None
        input_29 = torch.conv2d(
            input_28,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_28 = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_30 = torch.nn.functional.batch_norm(
            input_29,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_29 = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_5 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_5 = None
        input_30 += result_4
        result_5 = input_30
        input_30 = result_4 = None
        input_31 = torch.conv2d(
            result_5,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_32 = torch.nn.functional.batch_norm(
            input_31,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_31 = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_33 = torch.nn.functional.silu(input_32, inplace=True)
        input_32 = None
        input_34 = torch.conv2d(
            input_33,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_33 = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_35 = torch.nn.functional.batch_norm(
            input_34,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_34 = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_6 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_6 = None
        input_35 += result_5
        result_6 = input_35
        input_35 = result_5 = None
        input_36 = torch.conv2d(
            result_6,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_37 = torch.nn.functional.batch_norm(
            input_36,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_36 = l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_38 = torch.nn.functional.silu(input_37, inplace=True)
        input_37 = None
        input_39 = torch.conv2d(
            input_38,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_38 = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_40 = torch.nn.functional.batch_norm(
            input_39,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_39 = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_7 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_7 = None
        input_40 += result_6
        result_7 = input_40
        input_40 = result_6 = None
        input_41 = torch.conv2d(
            result_7,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_42 = torch.nn.functional.batch_norm(
            input_41,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_41 = l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_43 = torch.nn.functional.silu(input_42, inplace=True)
        input_42 = None
        input_44 = torch.conv2d(
            input_43,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_43 = l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_45 = torch.nn.functional.batch_norm(
            input_44,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_44 = l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_8 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_8 = None
        input_45 += result_7
        result_8 = input_45
        input_45 = result_7 = None
        input_46 = torch.conv2d(
            result_8,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_47 = torch.nn.functional.batch_norm(
            input_46,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_46 = l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_48 = torch.nn.functional.silu(input_47, inplace=True)
        input_47 = None
        input_49 = torch.conv2d(
            input_48,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_48 = l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_50 = torch.nn.functional.batch_norm(
            input_49,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_49 = l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_9 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_9 = None
        input_50 += result_8
        result_9 = input_50
        input_50 = result_8 = None
        input_51 = torch.conv2d(
            result_9,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        result_9 = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_52 = torch.nn.functional.batch_norm(
            input_51,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_51 = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_53 = torch.nn.functional.silu(input_52, inplace=True)
        input_52 = None
        input_54 = torch.conv2d(
            input_53,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_53 = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_55 = torch.nn.functional.batch_norm(
            input_54,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_54 = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_56 = torch.conv2d(
            input_55,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_57 = torch.nn.functional.batch_norm(
            input_56,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_56 = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_58 = torch.nn.functional.silu(input_57, inplace=True)
        input_57 = None
        input_59 = torch.conv2d(
            input_58,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_58 = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_60 = torch.nn.functional.batch_norm(
            input_59,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_59 = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_10 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_10 = None
        input_60 += input_55
        result_10 = input_60
        input_60 = input_55 = None
        input_61 = torch.conv2d(
            result_10,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_62 = torch.nn.functional.batch_norm(
            input_61,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_61 = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_63 = torch.nn.functional.silu(input_62, inplace=True)
        input_62 = None
        input_64 = torch.conv2d(
            input_63,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_63 = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_65 = torch.nn.functional.batch_norm(
            input_64,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_64 = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_11 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_11 = None
        input_65 += result_10
        result_11 = input_65
        input_65 = result_10 = None
        input_66 = torch.conv2d(
            result_11,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_67 = torch.nn.functional.batch_norm(
            input_66,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_66 = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_68 = torch.nn.functional.silu(input_67, inplace=True)
        input_67 = None
        input_69 = torch.conv2d(
            input_68,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_68 = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_70 = torch.nn.functional.batch_norm(
            input_69,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_69 = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_12 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_12 = None
        input_70 += result_11
        result_12 = input_70
        input_70 = result_11 = None
        input_71 = torch.conv2d(
            result_12,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_72 = torch.nn.functional.batch_norm(
            input_71,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_71 = l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_73 = torch.nn.functional.silu(input_72, inplace=True)
        input_72 = None
        input_74 = torch.conv2d(
            input_73,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_73 = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_75 = torch.nn.functional.batch_norm(
            input_74,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_74 = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_13 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_13 = None
        input_75 += result_12
        result_13 = input_75
        input_75 = result_12 = None
        input_76 = torch.conv2d(
            result_13,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_77 = torch.nn.functional.batch_norm(
            input_76,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_76 = l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_78 = torch.nn.functional.silu(input_77, inplace=True)
        input_77 = None
        input_79 = torch.conv2d(
            input_78,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_78 = l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_80 = torch.nn.functional.batch_norm(
            input_79,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_79 = l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_14 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_14 = None
        input_80 += result_13
        result_14 = input_80
        input_80 = result_13 = None
        input_81 = torch.conv2d(
            result_14,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_82 = torch.nn.functional.batch_norm(
            input_81,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_81 = l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_83 = torch.nn.functional.silu(input_82, inplace=True)
        input_82 = None
        input_84 = torch.conv2d(
            input_83,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_83 = l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_85 = torch.nn.functional.batch_norm(
            input_84,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_84 = l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_15 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_15 = None
        input_85 += result_14
        result_15 = input_85
        input_85 = result_14 = None
        input_86 = torch.conv2d(
            result_15,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_15 = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_87 = torch.nn.functional.batch_norm(
            input_86,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_86 = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_88 = torch.nn.functional.silu(input_87, inplace=True)
        input_87 = None
        input_89 = torch.conv2d(
            input_88,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            384,
        )
        input_88 = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_90 = torch.nn.functional.batch_norm(
            input_89,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_89 = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_91 = torch.nn.functional.silu(input_90, inplace=True)
        input_90 = None
        scale = torch.nn.functional.adaptive_avg_pool2d(input_91, 1)
        scale_1 = torch.conv2d(
            scale,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale = l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_2 = torch.nn.functional.silu(scale_1, inplace=True)
        scale_1 = None
        scale_3 = torch.conv2d(
            scale_2,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_2 = l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_4 = torch.sigmoid(scale_3)
        scale_3 = None
        input_92 = scale_4 * input_91
        scale_4 = input_91 = None
        input_93 = torch.conv2d(
            input_92,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_92 = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_94 = torch.nn.functional.batch_norm(
            input_93,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_93 = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_95 = torch.conv2d(
            input_94,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_96 = torch.nn.functional.batch_norm(
            input_95,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_95 = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_97 = torch.nn.functional.silu(input_96, inplace=True)
        input_96 = None
        input_98 = torch.conv2d(
            input_97,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_97 = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_99 = torch.nn.functional.batch_norm(
            input_98,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_98 = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_100 = torch.nn.functional.silu(input_99, inplace=True)
        input_99 = None
        scale_5 = torch.nn.functional.adaptive_avg_pool2d(input_100, 1)
        scale_6 = torch.conv2d(
            scale_5,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_5 = l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_7 = torch.nn.functional.silu(scale_6, inplace=True)
        scale_6 = None
        scale_8 = torch.conv2d(
            scale_7,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_7 = l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_9 = torch.sigmoid(scale_8)
        scale_8 = None
        input_101 = scale_9 * input_100
        scale_9 = input_100 = None
        input_102 = torch.conv2d(
            input_101,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_101 = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_103 = torch.nn.functional.batch_norm(
            input_102,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_102 = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_16 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_16 = None
        input_103 += input_94
        result_16 = input_103
        input_103 = input_94 = None
        input_104 = torch.conv2d(
            result_16,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_105 = torch.nn.functional.batch_norm(
            input_104,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_104 = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_106 = torch.nn.functional.silu(input_105, inplace=True)
        input_105 = None
        input_107 = torch.conv2d(
            input_106,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_106 = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_108 = torch.nn.functional.batch_norm(
            input_107,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_107 = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_109 = torch.nn.functional.silu(input_108, inplace=True)
        input_108 = None
        scale_10 = torch.nn.functional.adaptive_avg_pool2d(input_109, 1)
        scale_11 = torch.conv2d(
            scale_10,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_10 = l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_12 = torch.nn.functional.silu(scale_11, inplace=True)
        scale_11 = None
        scale_13 = torch.conv2d(
            scale_12,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_12 = l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_14 = torch.sigmoid(scale_13)
        scale_13 = None
        input_110 = scale_14 * input_109
        scale_14 = input_109 = None
        input_111 = torch.conv2d(
            input_110,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_110 = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_112 = torch.nn.functional.batch_norm(
            input_111,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_111 = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_17 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_17 = None
        input_112 += result_16
        result_17 = input_112
        input_112 = result_16 = None
        input_113 = torch.conv2d(
            result_17,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_114 = torch.nn.functional.batch_norm(
            input_113,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_113 = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_115 = torch.nn.functional.silu(input_114, inplace=True)
        input_114 = None
        input_116 = torch.conv2d(
            input_115,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_115 = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_117 = torch.nn.functional.batch_norm(
            input_116,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_116 = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_118 = torch.nn.functional.silu(input_117, inplace=True)
        input_117 = None
        scale_15 = torch.nn.functional.adaptive_avg_pool2d(input_118, 1)
        scale_16 = torch.conv2d(
            scale_15,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_15 = l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_17 = torch.nn.functional.silu(scale_16, inplace=True)
        scale_16 = None
        scale_18 = torch.conv2d(
            scale_17,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_17 = l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_19 = torch.sigmoid(scale_18)
        scale_18 = None
        input_119 = scale_19 * input_118
        scale_19 = input_118 = None
        input_120 = torch.conv2d(
            input_119,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_119 = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_121 = torch.nn.functional.batch_norm(
            input_120,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_120 = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_18 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_18 = None
        input_121 += result_17
        result_18 = input_121
        input_121 = result_17 = None
        input_122 = torch.conv2d(
            result_18,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_123 = torch.nn.functional.batch_norm(
            input_122,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_122 = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_124 = torch.nn.functional.silu(input_123, inplace=True)
        input_123 = None
        input_125 = torch.conv2d(
            input_124,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_124 = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_126 = torch.nn.functional.batch_norm(
            input_125,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_125 = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_127 = torch.nn.functional.silu(input_126, inplace=True)
        input_126 = None
        scale_20 = torch.nn.functional.adaptive_avg_pool2d(input_127, 1)
        scale_21 = torch.conv2d(
            scale_20,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_20 = l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_22 = torch.nn.functional.silu(scale_21, inplace=True)
        scale_21 = None
        scale_23 = torch.conv2d(
            scale_22,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_22 = l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_24 = torch.sigmoid(scale_23)
        scale_23 = None
        input_128 = scale_24 * input_127
        scale_24 = input_127 = None
        input_129 = torch.conv2d(
            input_128,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_128 = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_130 = torch.nn.functional.batch_norm(
            input_129,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_129 = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_19 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_19 = None
        input_130 += result_18
        result_19 = input_130
        input_130 = result_18 = None
        input_131 = torch.conv2d(
            result_19,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_132 = torch.nn.functional.batch_norm(
            input_131,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_131 = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_133 = torch.nn.functional.silu(input_132, inplace=True)
        input_132 = None
        input_134 = torch.conv2d(
            input_133,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_133 = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_135 = torch.nn.functional.batch_norm(
            input_134,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_134 = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_136 = torch.nn.functional.silu(input_135, inplace=True)
        input_135 = None
        scale_25 = torch.nn.functional.adaptive_avg_pool2d(input_136, 1)
        scale_26 = torch.conv2d(
            scale_25,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_25 = l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_27 = torch.nn.functional.silu(scale_26, inplace=True)
        scale_26 = None
        scale_28 = torch.conv2d(
            scale_27,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_27 = l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_29 = torch.sigmoid(scale_28)
        scale_28 = None
        input_137 = scale_29 * input_136
        scale_29 = input_136 = None
        input_138 = torch.conv2d(
            input_137,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_137 = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_139 = torch.nn.functional.batch_norm(
            input_138,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_138 = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_20 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_20 = None
        input_139 += result_19
        result_20 = input_139
        input_139 = result_19 = None
        input_140 = torch.conv2d(
            result_20,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_141 = torch.nn.functional.batch_norm(
            input_140,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_140 = l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_142 = torch.nn.functional.silu(input_141, inplace=True)
        input_141 = None
        input_143 = torch.conv2d(
            input_142,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_142 = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_144 = torch.nn.functional.batch_norm(
            input_143,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_143 = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_145 = torch.nn.functional.silu(input_144, inplace=True)
        input_144 = None
        scale_30 = torch.nn.functional.adaptive_avg_pool2d(input_145, 1)
        scale_31 = torch.conv2d(
            scale_30,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_30 = l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_32 = torch.nn.functional.silu(scale_31, inplace=True)
        scale_31 = None
        scale_33 = torch.conv2d(
            scale_32,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_32 = l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_34 = torch.sigmoid(scale_33)
        scale_33 = None
        input_146 = scale_34 * input_145
        scale_34 = input_145 = None
        input_147 = torch.conv2d(
            input_146,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_146 = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_148 = torch.nn.functional.batch_norm(
            input_147,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_147 = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_21 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_21 = None
        input_148 += result_20
        result_21 = input_148
        input_148 = result_20 = None
        input_149 = torch.conv2d(
            result_21,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_150 = torch.nn.functional.batch_norm(
            input_149,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_149 = l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_151 = torch.nn.functional.silu(input_150, inplace=True)
        input_150 = None
        input_152 = torch.conv2d(
            input_151,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_151 = l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_153 = torch.nn.functional.batch_norm(
            input_152,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_152 = l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_154 = torch.nn.functional.silu(input_153, inplace=True)
        input_153 = None
        scale_35 = torch.nn.functional.adaptive_avg_pool2d(input_154, 1)
        scale_36 = torch.conv2d(
            scale_35,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_35 = l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_37 = torch.nn.functional.silu(scale_36, inplace=True)
        scale_36 = None
        scale_38 = torch.conv2d(
            scale_37,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_37 = l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_39 = torch.sigmoid(scale_38)
        scale_38 = None
        input_155 = scale_39 * input_154
        scale_39 = input_154 = None
        input_156 = torch.conv2d(
            input_155,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_155 = l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_157 = torch.nn.functional.batch_norm(
            input_156,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_156 = l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_22 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_22 = None
        input_157 += result_21
        result_22 = input_157
        input_157 = result_21 = None
        input_158 = torch.conv2d(
            result_22,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_159 = torch.nn.functional.batch_norm(
            input_158,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_158 = l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_160 = torch.nn.functional.silu(input_159, inplace=True)
        input_159 = None
        input_161 = torch.conv2d(
            input_160,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_160 = l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_162 = torch.nn.functional.batch_norm(
            input_161,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_161 = l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_163 = torch.nn.functional.silu(input_162, inplace=True)
        input_162 = None
        scale_40 = torch.nn.functional.adaptive_avg_pool2d(input_163, 1)
        scale_41 = torch.conv2d(
            scale_40,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_40 = l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_42 = torch.nn.functional.silu(scale_41, inplace=True)
        scale_41 = None
        scale_43 = torch.conv2d(
            scale_42,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_42 = l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_44 = torch.sigmoid(scale_43)
        scale_43 = None
        input_164 = scale_44 * input_163
        scale_44 = input_163 = None
        input_165 = torch.conv2d(
            input_164,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_164 = l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_166 = torch.nn.functional.batch_norm(
            input_165,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_165 = l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_23 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_23 = None
        input_166 += result_22
        result_23 = input_166
        input_166 = result_22 = None
        input_167 = torch.conv2d(
            result_23,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_168 = torch.nn.functional.batch_norm(
            input_167,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_167 = l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_169 = torch.nn.functional.silu(input_168, inplace=True)
        input_168 = None
        input_170 = torch.conv2d(
            input_169,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_169 = l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_171 = torch.nn.functional.batch_norm(
            input_170,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_170 = l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_172 = torch.nn.functional.silu(input_171, inplace=True)
        input_171 = None
        scale_45 = torch.nn.functional.adaptive_avg_pool2d(input_172, 1)
        scale_46 = torch.conv2d(
            scale_45,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_45 = l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_47 = torch.nn.functional.silu(scale_46, inplace=True)
        scale_46 = None
        scale_48 = torch.conv2d(
            scale_47,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_47 = l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_49 = torch.sigmoid(scale_48)
        scale_48 = None
        input_173 = scale_49 * input_172
        scale_49 = input_172 = None
        input_174 = torch.conv2d(
            input_173,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_173 = l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_175 = torch.nn.functional.batch_norm(
            input_174,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_174 = l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_24 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_24 = None
        input_175 += result_23
        result_24 = input_175
        input_175 = result_23 = None
        input_176 = torch.conv2d(
            result_24,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_24 = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_177 = torch.nn.functional.batch_norm(
            input_176,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_176 = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_178 = torch.nn.functional.silu(input_177, inplace=True)
        input_177 = None
        input_179 = torch.conv2d(
            input_178,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1152,
        )
        input_178 = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_180 = torch.nn.functional.batch_norm(
            input_179,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_179 = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_181 = torch.nn.functional.silu(input_180, inplace=True)
        input_180 = None
        scale_50 = torch.nn.functional.adaptive_avg_pool2d(input_181, 1)
        scale_51 = torch.conv2d(
            scale_50,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_50 = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_52 = torch.nn.functional.silu(scale_51, inplace=True)
        scale_51 = None
        scale_53 = torch.conv2d(
            scale_52,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_52 = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_54 = torch.sigmoid(scale_53)
        scale_53 = None
        input_182 = scale_54 * input_181
        scale_54 = input_181 = None
        input_183 = torch.conv2d(
            input_182,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_182 = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_184 = torch.nn.functional.batch_norm(
            input_183,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_183 = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_185 = torch.conv2d(
            input_184,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_186 = torch.nn.functional.batch_norm(
            input_185,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_185 = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_187 = torch.nn.functional.silu(input_186, inplace=True)
        input_186 = None
        input_188 = torch.conv2d(
            input_187,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1344,
        )
        input_187 = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_189 = torch.nn.functional.batch_norm(
            input_188,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_188 = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_190 = torch.nn.functional.silu(input_189, inplace=True)
        input_189 = None
        scale_55 = torch.nn.functional.adaptive_avg_pool2d(input_190, 1)
        scale_56 = torch.conv2d(
            scale_55,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_55 = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_57 = torch.nn.functional.silu(scale_56, inplace=True)
        scale_56 = None
        scale_58 = torch.conv2d(
            scale_57,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_57 = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_59 = torch.sigmoid(scale_58)
        scale_58 = None
        input_191 = scale_59 * input_190
        scale_59 = input_190 = None
        input_192 = torch.conv2d(
            input_191,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_191 = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_193 = torch.nn.functional.batch_norm(
            input_192,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_192 = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_25 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_25 = None
        input_193 += input_184
        result_25 = input_193
        input_193 = input_184 = None
        input_194 = torch.conv2d(
            result_25,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_195 = torch.nn.functional.batch_norm(
            input_194,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_194 = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_196 = torch.nn.functional.silu(input_195, inplace=True)
        input_195 = None
        input_197 = torch.conv2d(
            input_196,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1344,
        )
        input_196 = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_198 = torch.nn.functional.batch_norm(
            input_197,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_197 = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_199 = torch.nn.functional.silu(input_198, inplace=True)
        input_198 = None
        scale_60 = torch.nn.functional.adaptive_avg_pool2d(input_199, 1)
        scale_61 = torch.conv2d(
            scale_60,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_60 = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_62 = torch.nn.functional.silu(scale_61, inplace=True)
        scale_61 = None
        scale_63 = torch.conv2d(
            scale_62,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_62 = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_64 = torch.sigmoid(scale_63)
        scale_63 = None
        input_200 = scale_64 * input_199
        scale_64 = input_199 = None
        input_201 = torch.conv2d(
            input_200,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_200 = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_202 = torch.nn.functional.batch_norm(
            input_201,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_201 = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_26 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_26 = None
        input_202 += result_25
        result_26 = input_202
        input_202 = result_25 = None
        input_203 = torch.conv2d(
            result_26,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_204 = torch.nn.functional.batch_norm(
            input_203,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_203 = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_205 = torch.nn.functional.silu(input_204, inplace=True)
        input_204 = None
        input_206 = torch.conv2d(
            input_205,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1344,
        )
        input_205 = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_207 = torch.nn.functional.batch_norm(
            input_206,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_206 = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_208 = torch.nn.functional.silu(input_207, inplace=True)
        input_207 = None
        scale_65 = torch.nn.functional.adaptive_avg_pool2d(input_208, 1)
        scale_66 = torch.conv2d(
            scale_65,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_65 = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_67 = torch.nn.functional.silu(scale_66, inplace=True)
        scale_66 = None
        scale_68 = torch.conv2d(
            scale_67,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_67 = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_69 = torch.sigmoid(scale_68)
        scale_68 = None
        input_209 = scale_69 * input_208
        scale_69 = input_208 = None
        input_210 = torch.conv2d(
            input_209,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_209 = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_211 = torch.nn.functional.batch_norm(
            input_210,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_210 = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_27 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_27 = None
        input_211 += result_26
        result_27 = input_211
        input_211 = result_26 = None
        input_212 = torch.conv2d(
            result_27,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_213 = torch.nn.functional.batch_norm(
            input_212,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_212 = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_214 = torch.nn.functional.silu(input_213, inplace=True)
        input_213 = None
        input_215 = torch.conv2d(
            input_214,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1344,
        )
        input_214 = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_216 = torch.nn.functional.batch_norm(
            input_215,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_215 = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_217 = torch.nn.functional.silu(input_216, inplace=True)
        input_216 = None
        scale_70 = torch.nn.functional.adaptive_avg_pool2d(input_217, 1)
        scale_71 = torch.conv2d(
            scale_70,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_70 = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_72 = torch.nn.functional.silu(scale_71, inplace=True)
        scale_71 = None
        scale_73 = torch.conv2d(
            scale_72,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_72 = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_74 = torch.sigmoid(scale_73)
        scale_73 = None
        input_218 = scale_74 * input_217
        scale_74 = input_217 = None
        input_219 = torch.conv2d(
            input_218,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_218 = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_220 = torch.nn.functional.batch_norm(
            input_219,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_219 = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_28 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_28 = None
        input_220 += result_27
        result_28 = input_220
        input_220 = result_27 = None
        input_221 = torch.conv2d(
            result_28,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_222 = torch.nn.functional.batch_norm(
            input_221,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_221 = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_223 = torch.nn.functional.silu(input_222, inplace=True)
        input_222 = None
        input_224 = torch.conv2d(
            input_223,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1344,
        )
        input_223 = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_225 = torch.nn.functional.batch_norm(
            input_224,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_224 = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_226 = torch.nn.functional.silu(input_225, inplace=True)
        input_225 = None
        scale_75 = torch.nn.functional.adaptive_avg_pool2d(input_226, 1)
        scale_76 = torch.conv2d(
            scale_75,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_75 = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_77 = torch.nn.functional.silu(scale_76, inplace=True)
        scale_76 = None
        scale_78 = torch.conv2d(
            scale_77,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_77 = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_79 = torch.sigmoid(scale_78)
        scale_78 = None
        input_227 = scale_79 * input_226
        scale_79 = input_226 = None
        input_228 = torch.conv2d(
            input_227,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_227 = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_229 = torch.nn.functional.batch_norm(
            input_228,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_228 = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_29 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_29 = None
        input_229 += result_28
        result_29 = input_229
        input_229 = result_28 = None
        input_230 = torch.conv2d(
            result_29,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_231 = torch.nn.functional.batch_norm(
            input_230,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_230 = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_232 = torch.nn.functional.silu(input_231, inplace=True)
        input_231 = None
        input_233 = torch.conv2d(
            input_232,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1344,
        )
        input_232 = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_234 = torch.nn.functional.batch_norm(
            input_233,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_233 = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_235 = torch.nn.functional.silu(input_234, inplace=True)
        input_234 = None
        scale_80 = torch.nn.functional.adaptive_avg_pool2d(input_235, 1)
        scale_81 = torch.conv2d(
            scale_80,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_80 = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_82 = torch.nn.functional.silu(scale_81, inplace=True)
        scale_81 = None
        scale_83 = torch.conv2d(
            scale_82,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_82 = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_84 = torch.sigmoid(scale_83)
        scale_83 = None
        input_236 = scale_84 * input_235
        scale_84 = input_235 = None
        input_237 = torch.conv2d(
            input_236,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_236 = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_238 = torch.nn.functional.batch_norm(
            input_237,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_237 = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_30 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_30 = None
        input_238 += result_29
        result_30 = input_238
        input_238 = result_29 = None
        input_239 = torch.conv2d(
            result_30,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_240 = torch.nn.functional.batch_norm(
            input_239,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_239 = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_241 = torch.nn.functional.silu(input_240, inplace=True)
        input_240 = None
        input_242 = torch.conv2d(
            input_241,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1344,
        )
        input_241 = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_243 = torch.nn.functional.batch_norm(
            input_242,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_242 = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_244 = torch.nn.functional.silu(input_243, inplace=True)
        input_243 = None
        scale_85 = torch.nn.functional.adaptive_avg_pool2d(input_244, 1)
        scale_86 = torch.conv2d(
            scale_85,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_85 = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_87 = torch.nn.functional.silu(scale_86, inplace=True)
        scale_86 = None
        scale_88 = torch.conv2d(
            scale_87,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_87 = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_89 = torch.sigmoid(scale_88)
        scale_88 = None
        input_245 = scale_89 * input_244
        scale_89 = input_244 = None
        input_246 = torch.conv2d(
            input_245,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_245 = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_247 = torch.nn.functional.batch_norm(
            input_246,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_246 = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_31 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_31 = None
        input_247 += result_30
        result_31 = input_247
        input_247 = result_30 = None
        input_248 = torch.conv2d(
            result_31,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_249 = torch.nn.functional.batch_norm(
            input_248,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_248 = l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_250 = torch.nn.functional.silu(input_249, inplace=True)
        input_249 = None
        input_251 = torch.conv2d(
            input_250,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1344,
        )
        input_250 = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_252 = torch.nn.functional.batch_norm(
            input_251,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_251 = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_253 = torch.nn.functional.silu(input_252, inplace=True)
        input_252 = None
        scale_90 = torch.nn.functional.adaptive_avg_pool2d(input_253, 1)
        scale_91 = torch.conv2d(
            scale_90,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_90 = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_92 = torch.nn.functional.silu(scale_91, inplace=True)
        scale_91 = None
        scale_93 = torch.conv2d(
            scale_92,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_92 = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_94 = torch.sigmoid(scale_93)
        scale_93 = None
        input_254 = scale_94 * input_253
        scale_94 = input_253 = None
        input_255 = torch.conv2d(
            input_254,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_254 = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_256 = torch.nn.functional.batch_norm(
            input_255,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_255 = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_32 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_32 = None
        input_256 += result_31
        result_32 = input_256
        input_256 = result_31 = None
        input_257 = torch.conv2d(
            result_32,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_258 = torch.nn.functional.batch_norm(
            input_257,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_257 = l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_259 = torch.nn.functional.silu(input_258, inplace=True)
        input_258 = None
        input_260 = torch.conv2d(
            input_259,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1344,
        )
        input_259 = l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_261 = torch.nn.functional.batch_norm(
            input_260,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_260 = l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_262 = torch.nn.functional.silu(input_261, inplace=True)
        input_261 = None
        scale_95 = torch.nn.functional.adaptive_avg_pool2d(input_262, 1)
        scale_96 = torch.conv2d(
            scale_95,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_95 = l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_97 = torch.nn.functional.silu(scale_96, inplace=True)
        scale_96 = None
        scale_98 = torch.conv2d(
            scale_97,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_97 = l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_99 = torch.sigmoid(scale_98)
        scale_98 = None
        input_263 = scale_99 * input_262
        scale_99 = input_262 = None
        input_264 = torch.conv2d(
            input_263,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_263 = l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_265 = torch.nn.functional.batch_norm(
            input_264,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_264 = l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_33 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_33 = None
        input_265 += result_32
        result_33 = input_265
        input_265 = result_32 = None
        input_266 = torch.conv2d(
            result_33,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_267 = torch.nn.functional.batch_norm(
            input_266,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_266 = l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_268 = torch.nn.functional.silu(input_267, inplace=True)
        input_267 = None
        input_269 = torch.conv2d(
            input_268,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1344,
        )
        input_268 = l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_270 = torch.nn.functional.batch_norm(
            input_269,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_269 = l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_271 = torch.nn.functional.silu(input_270, inplace=True)
        input_270 = None
        scale_100 = torch.nn.functional.adaptive_avg_pool2d(input_271, 1)
        scale_101 = torch.conv2d(
            scale_100,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_100 = l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_102 = torch.nn.functional.silu(scale_101, inplace=True)
        scale_101 = None
        scale_103 = torch.conv2d(
            scale_102,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_102 = l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_104 = torch.sigmoid(scale_103)
        scale_103 = None
        input_272 = scale_104 * input_271
        scale_104 = input_271 = None
        input_273 = torch.conv2d(
            input_272,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_272 = l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_274 = torch.nn.functional.batch_norm(
            input_273,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_273 = l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_34 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_34 = None
        input_274 += result_33
        result_34 = input_274
        input_274 = result_33 = None
        input_275 = torch.conv2d(
            result_34,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_276 = torch.nn.functional.batch_norm(
            input_275,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_275 = l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_277 = torch.nn.functional.silu(input_276, inplace=True)
        input_276 = None
        input_278 = torch.conv2d(
            input_277,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1344,
        )
        input_277 = l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_279 = torch.nn.functional.batch_norm(
            input_278,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_278 = l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_280 = torch.nn.functional.silu(input_279, inplace=True)
        input_279 = None
        scale_105 = torch.nn.functional.adaptive_avg_pool2d(input_280, 1)
        scale_106 = torch.conv2d(
            scale_105,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_105 = l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_107 = torch.nn.functional.silu(scale_106, inplace=True)
        scale_106 = None
        scale_108 = torch.conv2d(
            scale_107,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_107 = l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_109 = torch.sigmoid(scale_108)
        scale_108 = None
        input_281 = scale_109 * input_280
        scale_109 = input_280 = None
        input_282 = torch.conv2d(
            input_281,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_281 = l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_283 = torch.nn.functional.batch_norm(
            input_282,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_282 = l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_35 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_35 = None
        input_283 += result_34
        result_35 = input_283
        input_283 = result_34 = None
        input_284 = torch.conv2d(
            result_35,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_285 = torch.nn.functional.batch_norm(
            input_284,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_284 = l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_286 = torch.nn.functional.silu(input_285, inplace=True)
        input_285 = None
        input_287 = torch.conv2d(
            input_286,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1344,
        )
        input_286 = l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_288 = torch.nn.functional.batch_norm(
            input_287,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_287 = l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_289 = torch.nn.functional.silu(input_288, inplace=True)
        input_288 = None
        scale_110 = torch.nn.functional.adaptive_avg_pool2d(input_289, 1)
        scale_111 = torch.conv2d(
            scale_110,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_110 = l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_112 = torch.nn.functional.silu(scale_111, inplace=True)
        scale_111 = None
        scale_113 = torch.conv2d(
            scale_112,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_112 = l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_114 = torch.sigmoid(scale_113)
        scale_113 = None
        input_290 = scale_114 * input_289
        scale_114 = input_289 = None
        input_291 = torch.conv2d(
            input_290,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_290 = l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_292 = torch.nn.functional.batch_norm(
            input_291,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_291 = l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_36 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_36 = None
        input_292 += result_35
        result_36 = input_292
        input_292 = result_35 = None
        input_293 = torch.conv2d(
            result_36,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_294 = torch.nn.functional.batch_norm(
            input_293,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_293 = l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_295 = torch.nn.functional.silu(input_294, inplace=True)
        input_294 = None
        input_296 = torch.conv2d(
            input_295,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1344,
        )
        input_295 = l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_297 = torch.nn.functional.batch_norm(
            input_296,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_296 = l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_298 = torch.nn.functional.silu(input_297, inplace=True)
        input_297 = None
        scale_115 = torch.nn.functional.adaptive_avg_pool2d(input_298, 1)
        scale_116 = torch.conv2d(
            scale_115,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_115 = l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_117 = torch.nn.functional.silu(scale_116, inplace=True)
        scale_116 = None
        scale_118 = torch.conv2d(
            scale_117,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_117 = l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_119 = torch.sigmoid(scale_118)
        scale_118 = None
        input_299 = scale_119 * input_298
        scale_119 = input_298 = None
        input_300 = torch.conv2d(
            input_299,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_299 = l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_301 = torch.nn.functional.batch_norm(
            input_300,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_300 = l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_37 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_37 = None
        input_301 += result_36
        result_37 = input_301
        input_301 = result_36 = None
        input_302 = torch.conv2d(
            result_37,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_303 = torch.nn.functional.batch_norm(
            input_302,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_302 = l_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_14_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_304 = torch.nn.functional.silu(input_303, inplace=True)
        input_303 = None
        input_305 = torch.conv2d(
            input_304,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1344,
        )
        input_304 = l_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_306 = torch.nn.functional.batch_norm(
            input_305,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_305 = l_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_14_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_307 = torch.nn.functional.silu(input_306, inplace=True)
        input_306 = None
        scale_120 = torch.nn.functional.adaptive_avg_pool2d(input_307, 1)
        scale_121 = torch.conv2d(
            scale_120,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_120 = l_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_122 = torch.nn.functional.silu(scale_121, inplace=True)
        scale_121 = None
        scale_123 = torch.conv2d(
            scale_122,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_122 = l_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_124 = torch.sigmoid(scale_123)
        scale_123 = None
        input_308 = scale_124 * input_307
        scale_124 = input_307 = None
        input_309 = torch.conv2d(
            input_308,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_308 = l_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_310 = torch.nn.functional.batch_norm(
            input_309,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_309 = l_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_14_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_38 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_38 = None
        input_310 += result_37
        result_38 = input_310
        input_310 = result_37 = None
        input_311 = torch.conv2d(
            result_38,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_312 = torch.nn.functional.batch_norm(
            input_311,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_311 = l_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_15_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_313 = torch.nn.functional.silu(input_312, inplace=True)
        input_312 = None
        input_314 = torch.conv2d(
            input_313,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1344,
        )
        input_313 = l_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_315 = torch.nn.functional.batch_norm(
            input_314,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_314 = l_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_15_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_316 = torch.nn.functional.silu(input_315, inplace=True)
        input_315 = None
        scale_125 = torch.nn.functional.adaptive_avg_pool2d(input_316, 1)
        scale_126 = torch.conv2d(
            scale_125,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_125 = l_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_127 = torch.nn.functional.silu(scale_126, inplace=True)
        scale_126 = None
        scale_128 = torch.conv2d(
            scale_127,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_127 = l_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_15_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_129 = torch.sigmoid(scale_128)
        scale_128 = None
        input_317 = scale_129 * input_316
        scale_129 = input_316 = None
        input_318 = torch.conv2d(
            input_317,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_317 = l_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_319 = torch.nn.functional.batch_norm(
            input_318,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_318 = l_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_15_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_39 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_39 = None
        input_319 += result_38
        result_39 = input_319
        input_319 = result_38 = None
        input_320 = torch.conv2d(
            result_39,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_321 = torch.nn.functional.batch_norm(
            input_320,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_320 = l_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_16_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_322 = torch.nn.functional.silu(input_321, inplace=True)
        input_321 = None
        input_323 = torch.conv2d(
            input_322,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1344,
        )
        input_322 = l_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_324 = torch.nn.functional.batch_norm(
            input_323,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_323 = l_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_16_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_325 = torch.nn.functional.silu(input_324, inplace=True)
        input_324 = None
        scale_130 = torch.nn.functional.adaptive_avg_pool2d(input_325, 1)
        scale_131 = torch.conv2d(
            scale_130,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_130 = l_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_132 = torch.nn.functional.silu(scale_131, inplace=True)
        scale_131 = None
        scale_133 = torch.conv2d(
            scale_132,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_132 = l_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_16_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_134 = torch.sigmoid(scale_133)
        scale_133 = None
        input_326 = scale_134 * input_325
        scale_134 = input_325 = None
        input_327 = torch.conv2d(
            input_326,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_326 = l_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_328 = torch.nn.functional.batch_norm(
            input_327,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_327 = l_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_16_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_40 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_40 = None
        input_328 += result_39
        result_40 = input_328
        input_328 = result_39 = None
        input_329 = torch.conv2d(
            result_40,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_330 = torch.nn.functional.batch_norm(
            input_329,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_329 = l_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_17_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_331 = torch.nn.functional.silu(input_330, inplace=True)
        input_330 = None
        input_332 = torch.conv2d(
            input_331,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1344,
        )
        input_331 = l_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_333 = torch.nn.functional.batch_norm(
            input_332,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_332 = l_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_17_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_334 = torch.nn.functional.silu(input_333, inplace=True)
        input_333 = None
        scale_135 = torch.nn.functional.adaptive_avg_pool2d(input_334, 1)
        scale_136 = torch.conv2d(
            scale_135,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_135 = l_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_137 = torch.nn.functional.silu(scale_136, inplace=True)
        scale_136 = None
        scale_138 = torch.conv2d(
            scale_137,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_137 = l_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_17_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_139 = torch.sigmoid(scale_138)
        scale_138 = None
        input_335 = scale_139 * input_334
        scale_139 = input_334 = None
        input_336 = torch.conv2d(
            input_335,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_335 = l_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_337 = torch.nn.functional.batch_norm(
            input_336,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_336 = l_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_17_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_41 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_41 = None
        input_337 += result_40
        result_41 = input_337
        input_337 = result_40 = None
        input_338 = torch.conv2d(
            result_41,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_339 = torch.nn.functional.batch_norm(
            input_338,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_338 = l_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_18_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_340 = torch.nn.functional.silu(input_339, inplace=True)
        input_339 = None
        input_341 = torch.conv2d(
            input_340,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1344,
        )
        input_340 = l_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_342 = torch.nn.functional.batch_norm(
            input_341,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_341 = l_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_18_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_343 = torch.nn.functional.silu(input_342, inplace=True)
        input_342 = None
        scale_140 = torch.nn.functional.adaptive_avg_pool2d(input_343, 1)
        scale_141 = torch.conv2d(
            scale_140,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_140 = l_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_142 = torch.nn.functional.silu(scale_141, inplace=True)
        scale_141 = None
        scale_143 = torch.conv2d(
            scale_142,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_142 = l_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_18_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_144 = torch.sigmoid(scale_143)
        scale_143 = None
        input_344 = scale_144 * input_343
        scale_144 = input_343 = None
        input_345 = torch.conv2d(
            input_344,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_344 = l_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_346 = torch.nn.functional.batch_norm(
            input_345,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_345 = l_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_18_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_42 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_42 = None
        input_346 += result_41
        result_42 = input_346
        input_346 = result_41 = None
        input_347 = torch.conv2d(
            result_42,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_42 = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_348 = torch.nn.functional.batch_norm(
            input_347,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_347 = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_349 = torch.nn.functional.silu(input_348, inplace=True)
        input_348 = None
        input_350 = torch.conv2d(
            input_349,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1344,
        )
        input_349 = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_351 = torch.nn.functional.batch_norm(
            input_350,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_350 = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_352 = torch.nn.functional.silu(input_351, inplace=True)
        input_351 = None
        scale_145 = torch.nn.functional.adaptive_avg_pool2d(input_352, 1)
        scale_146 = torch.conv2d(
            scale_145,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_145 = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_147 = torch.nn.functional.silu(scale_146, inplace=True)
        scale_146 = None
        scale_148 = torch.conv2d(
            scale_147,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_147 = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_149 = torch.sigmoid(scale_148)
        scale_148 = None
        input_353 = scale_149 * input_352
        scale_149 = input_352 = None
        input_354 = torch.conv2d(
            input_353,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_353 = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_355 = torch.nn.functional.batch_norm(
            input_354,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_354 = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_356 = torch.conv2d(
            input_355,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_357 = torch.nn.functional.batch_norm(
            input_356,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_356 = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_358 = torch.nn.functional.silu(input_357, inplace=True)
        input_357 = None
        input_359 = torch.conv2d(
            input_358,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_358 = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_360 = torch.nn.functional.batch_norm(
            input_359,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_359 = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_361 = torch.nn.functional.silu(input_360, inplace=True)
        input_360 = None
        scale_150 = torch.nn.functional.adaptive_avg_pool2d(input_361, 1)
        scale_151 = torch.conv2d(
            scale_150,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_150 = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_152 = torch.nn.functional.silu(scale_151, inplace=True)
        scale_151 = None
        scale_153 = torch.conv2d(
            scale_152,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_152 = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_154 = torch.sigmoid(scale_153)
        scale_153 = None
        input_362 = scale_154 * input_361
        scale_154 = input_361 = None
        input_363 = torch.conv2d(
            input_362,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_362 = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_364 = torch.nn.functional.batch_norm(
            input_363,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_363 = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_43 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_43 = None
        input_364 += input_355
        result_43 = input_364
        input_364 = input_355 = None
        input_365 = torch.conv2d(
            result_43,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_366 = torch.nn.functional.batch_norm(
            input_365,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_365 = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_367 = torch.nn.functional.silu(input_366, inplace=True)
        input_366 = None
        input_368 = torch.conv2d(
            input_367,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_367 = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_369 = torch.nn.functional.batch_norm(
            input_368,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_368 = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_370 = torch.nn.functional.silu(input_369, inplace=True)
        input_369 = None
        scale_155 = torch.nn.functional.adaptive_avg_pool2d(input_370, 1)
        scale_156 = torch.conv2d(
            scale_155,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_155 = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_157 = torch.nn.functional.silu(scale_156, inplace=True)
        scale_156 = None
        scale_158 = torch.conv2d(
            scale_157,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_157 = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_159 = torch.sigmoid(scale_158)
        scale_158 = None
        input_371 = scale_159 * input_370
        scale_159 = input_370 = None
        input_372 = torch.conv2d(
            input_371,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_371 = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_373 = torch.nn.functional.batch_norm(
            input_372,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_372 = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_44 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_44 = None
        input_373 += result_43
        result_44 = input_373
        input_373 = result_43 = None
        input_374 = torch.conv2d(
            result_44,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_375 = torch.nn.functional.batch_norm(
            input_374,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_374 = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_376 = torch.nn.functional.silu(input_375, inplace=True)
        input_375 = None
        input_377 = torch.conv2d(
            input_376,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_376 = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_378 = torch.nn.functional.batch_norm(
            input_377,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_377 = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_379 = torch.nn.functional.silu(input_378, inplace=True)
        input_378 = None
        scale_160 = torch.nn.functional.adaptive_avg_pool2d(input_379, 1)
        scale_161 = torch.conv2d(
            scale_160,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_160 = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_162 = torch.nn.functional.silu(scale_161, inplace=True)
        scale_161 = None
        scale_163 = torch.conv2d(
            scale_162,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_162 = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_164 = torch.sigmoid(scale_163)
        scale_163 = None
        input_380 = scale_164 * input_379
        scale_164 = input_379 = None
        input_381 = torch.conv2d(
            input_380,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_380 = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_382 = torch.nn.functional.batch_norm(
            input_381,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_381 = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_45 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_45 = None
        input_382 += result_44
        result_45 = input_382
        input_382 = result_44 = None
        input_383 = torch.conv2d(
            result_45,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_384 = torch.nn.functional.batch_norm(
            input_383,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_383 = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_385 = torch.nn.functional.silu(input_384, inplace=True)
        input_384 = None
        input_386 = torch.conv2d(
            input_385,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_385 = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_387 = torch.nn.functional.batch_norm(
            input_386,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_386 = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_388 = torch.nn.functional.silu(input_387, inplace=True)
        input_387 = None
        scale_165 = torch.nn.functional.adaptive_avg_pool2d(input_388, 1)
        scale_166 = torch.conv2d(
            scale_165,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_165 = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_167 = torch.nn.functional.silu(scale_166, inplace=True)
        scale_166 = None
        scale_168 = torch.conv2d(
            scale_167,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_167 = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_169 = torch.sigmoid(scale_168)
        scale_168 = None
        input_389 = scale_169 * input_388
        scale_169 = input_388 = None
        input_390 = torch.conv2d(
            input_389,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_389 = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_391 = torch.nn.functional.batch_norm(
            input_390,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_390 = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_46 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_46 = None
        input_391 += result_45
        result_46 = input_391
        input_391 = result_45 = None
        input_392 = torch.conv2d(
            result_46,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_393 = torch.nn.functional.batch_norm(
            input_392,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_392 = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_394 = torch.nn.functional.silu(input_393, inplace=True)
        input_393 = None
        input_395 = torch.conv2d(
            input_394,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_394 = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_396 = torch.nn.functional.batch_norm(
            input_395,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_395 = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_397 = torch.nn.functional.silu(input_396, inplace=True)
        input_396 = None
        scale_170 = torch.nn.functional.adaptive_avg_pool2d(input_397, 1)
        scale_171 = torch.conv2d(
            scale_170,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_170 = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_172 = torch.nn.functional.silu(scale_171, inplace=True)
        scale_171 = None
        scale_173 = torch.conv2d(
            scale_172,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_172 = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_174 = torch.sigmoid(scale_173)
        scale_173 = None
        input_398 = scale_174 * input_397
        scale_174 = input_397 = None
        input_399 = torch.conv2d(
            input_398,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_398 = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_400 = torch.nn.functional.batch_norm(
            input_399,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_399 = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_47 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_47 = None
        input_400 += result_46
        result_47 = input_400
        input_400 = result_46 = None
        input_401 = torch.conv2d(
            result_47,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_402 = torch.nn.functional.batch_norm(
            input_401,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_401 = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_403 = torch.nn.functional.silu(input_402, inplace=True)
        input_402 = None
        input_404 = torch.conv2d(
            input_403,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_403 = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_405 = torch.nn.functional.batch_norm(
            input_404,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_404 = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_406 = torch.nn.functional.silu(input_405, inplace=True)
        input_405 = None
        scale_175 = torch.nn.functional.adaptive_avg_pool2d(input_406, 1)
        scale_176 = torch.conv2d(
            scale_175,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_175 = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_177 = torch.nn.functional.silu(scale_176, inplace=True)
        scale_176 = None
        scale_178 = torch.conv2d(
            scale_177,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_177 = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_179 = torch.sigmoid(scale_178)
        scale_178 = None
        input_407 = scale_179 * input_406
        scale_179 = input_406 = None
        input_408 = torch.conv2d(
            input_407,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_407 = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_409 = torch.nn.functional.batch_norm(
            input_408,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_408 = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_48 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_48 = None
        input_409 += result_47
        result_48 = input_409
        input_409 = result_47 = None
        input_410 = torch.conv2d(
            result_48,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_411 = torch.nn.functional.batch_norm(
            input_410,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_410 = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_412 = torch.nn.functional.silu(input_411, inplace=True)
        input_411 = None
        input_413 = torch.conv2d(
            input_412,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_412 = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_414 = torch.nn.functional.batch_norm(
            input_413,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_413 = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_415 = torch.nn.functional.silu(input_414, inplace=True)
        input_414 = None
        scale_180 = torch.nn.functional.adaptive_avg_pool2d(input_415, 1)
        scale_181 = torch.conv2d(
            scale_180,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_180 = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_182 = torch.nn.functional.silu(scale_181, inplace=True)
        scale_181 = None
        scale_183 = torch.conv2d(
            scale_182,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_182 = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_184 = torch.sigmoid(scale_183)
        scale_183 = None
        input_416 = scale_184 * input_415
        scale_184 = input_415 = None
        input_417 = torch.conv2d(
            input_416,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_416 = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_418 = torch.nn.functional.batch_norm(
            input_417,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_417 = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_49 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_49 = None
        input_418 += result_48
        result_49 = input_418
        input_418 = result_48 = None
        input_419 = torch.conv2d(
            result_49,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_420 = torch.nn.functional.batch_norm(
            input_419,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_419 = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_421 = torch.nn.functional.silu(input_420, inplace=True)
        input_420 = None
        input_422 = torch.conv2d(
            input_421,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_421 = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_423 = torch.nn.functional.batch_norm(
            input_422,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_422 = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_424 = torch.nn.functional.silu(input_423, inplace=True)
        input_423 = None
        scale_185 = torch.nn.functional.adaptive_avg_pool2d(input_424, 1)
        scale_186 = torch.conv2d(
            scale_185,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_185 = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_187 = torch.nn.functional.silu(scale_186, inplace=True)
        scale_186 = None
        scale_188 = torch.conv2d(
            scale_187,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_187 = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_189 = torch.sigmoid(scale_188)
        scale_188 = None
        input_425 = scale_189 * input_424
        scale_189 = input_424 = None
        input_426 = torch.conv2d(
            input_425,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_425 = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_427 = torch.nn.functional.batch_norm(
            input_426,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_426 = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_50 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_50 = None
        input_427 += result_49
        result_50 = input_427
        input_427 = result_49 = None
        input_428 = torch.conv2d(
            result_50,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_429 = torch.nn.functional.batch_norm(
            input_428,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_428 = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_430 = torch.nn.functional.silu(input_429, inplace=True)
        input_429 = None
        input_431 = torch.conv2d(
            input_430,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_430 = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_432 = torch.nn.functional.batch_norm(
            input_431,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_431 = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_433 = torch.nn.functional.silu(input_432, inplace=True)
        input_432 = None
        scale_190 = torch.nn.functional.adaptive_avg_pool2d(input_433, 1)
        scale_191 = torch.conv2d(
            scale_190,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_190 = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_192 = torch.nn.functional.silu(scale_191, inplace=True)
        scale_191 = None
        scale_193 = torch.conv2d(
            scale_192,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_192 = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_194 = torch.sigmoid(scale_193)
        scale_193 = None
        input_434 = scale_194 * input_433
        scale_194 = input_433 = None
        input_435 = torch.conv2d(
            input_434,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_434 = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_436 = torch.nn.functional.batch_norm(
            input_435,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_435 = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_51 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_51 = None
        input_436 += result_50
        result_51 = input_436
        input_436 = result_50 = None
        input_437 = torch.conv2d(
            result_51,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_438 = torch.nn.functional.batch_norm(
            input_437,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_437 = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_439 = torch.nn.functional.silu(input_438, inplace=True)
        input_438 = None
        input_440 = torch.conv2d(
            input_439,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_439 = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_441 = torch.nn.functional.batch_norm(
            input_440,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_440 = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_442 = torch.nn.functional.silu(input_441, inplace=True)
        input_441 = None
        scale_195 = torch.nn.functional.adaptive_avg_pool2d(input_442, 1)
        scale_196 = torch.conv2d(
            scale_195,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_195 = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_197 = torch.nn.functional.silu(scale_196, inplace=True)
        scale_196 = None
        scale_198 = torch.conv2d(
            scale_197,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_197 = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_199 = torch.sigmoid(scale_198)
        scale_198 = None
        input_443 = scale_199 * input_442
        scale_199 = input_442 = None
        input_444 = torch.conv2d(
            input_443,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_443 = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_445 = torch.nn.functional.batch_norm(
            input_444,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_444 = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_52 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_52 = None
        input_445 += result_51
        result_52 = input_445
        input_445 = result_51 = None
        input_446 = torch.conv2d(
            result_52,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_447 = torch.nn.functional.batch_norm(
            input_446,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_446 = l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_448 = torch.nn.functional.silu(input_447, inplace=True)
        input_447 = None
        input_449 = torch.conv2d(
            input_448,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_448 = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_450 = torch.nn.functional.batch_norm(
            input_449,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_449 = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_451 = torch.nn.functional.silu(input_450, inplace=True)
        input_450 = None
        scale_200 = torch.nn.functional.adaptive_avg_pool2d(input_451, 1)
        scale_201 = torch.conv2d(
            scale_200,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_200 = l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_202 = torch.nn.functional.silu(scale_201, inplace=True)
        scale_201 = None
        scale_203 = torch.conv2d(
            scale_202,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_202 = l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_204 = torch.sigmoid(scale_203)
        scale_203 = None
        input_452 = scale_204 * input_451
        scale_204 = input_451 = None
        input_453 = torch.conv2d(
            input_452,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_452 = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_454 = torch.nn.functional.batch_norm(
            input_453,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_453 = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_53 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_53 = None
        input_454 += result_52
        result_53 = input_454
        input_454 = result_52 = None
        input_455 = torch.conv2d(
            result_53,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_456 = torch.nn.functional.batch_norm(
            input_455,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_455 = l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_457 = torch.nn.functional.silu(input_456, inplace=True)
        input_456 = None
        input_458 = torch.conv2d(
            input_457,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_457 = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_459 = torch.nn.functional.batch_norm(
            input_458,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_458 = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_460 = torch.nn.functional.silu(input_459, inplace=True)
        input_459 = None
        scale_205 = torch.nn.functional.adaptive_avg_pool2d(input_460, 1)
        scale_206 = torch.conv2d(
            scale_205,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_205 = l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_207 = torch.nn.functional.silu(scale_206, inplace=True)
        scale_206 = None
        scale_208 = torch.conv2d(
            scale_207,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_207 = l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_209 = torch.sigmoid(scale_208)
        scale_208 = None
        input_461 = scale_209 * input_460
        scale_209 = input_460 = None
        input_462 = torch.conv2d(
            input_461,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_461 = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_463 = torch.nn.functional.batch_norm(
            input_462,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_462 = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_54 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_54 = None
        input_463 += result_53
        result_54 = input_463
        input_463 = result_53 = None
        input_464 = torch.conv2d(
            result_54,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_465 = torch.nn.functional.batch_norm(
            input_464,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_464 = l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_466 = torch.nn.functional.silu(input_465, inplace=True)
        input_465 = None
        input_467 = torch.conv2d(
            input_466,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_466 = l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_468 = torch.nn.functional.batch_norm(
            input_467,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_467 = l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_469 = torch.nn.functional.silu(input_468, inplace=True)
        input_468 = None
        scale_210 = torch.nn.functional.adaptive_avg_pool2d(input_469, 1)
        scale_211 = torch.conv2d(
            scale_210,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_210 = l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_212 = torch.nn.functional.silu(scale_211, inplace=True)
        scale_211 = None
        scale_213 = torch.conv2d(
            scale_212,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_212 = l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_214 = torch.sigmoid(scale_213)
        scale_213 = None
        input_470 = scale_214 * input_469
        scale_214 = input_469 = None
        input_471 = torch.conv2d(
            input_470,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_470 = l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_472 = torch.nn.functional.batch_norm(
            input_471,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_471 = l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_55 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_55 = None
        input_472 += result_54
        result_55 = input_472
        input_472 = result_54 = None
        input_473 = torch.conv2d(
            result_55,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_474 = torch.nn.functional.batch_norm(
            input_473,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_473 = l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_475 = torch.nn.functional.silu(input_474, inplace=True)
        input_474 = None
        input_476 = torch.conv2d(
            input_475,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_475 = l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_477 = torch.nn.functional.batch_norm(
            input_476,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_476 = l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_478 = torch.nn.functional.silu(input_477, inplace=True)
        input_477 = None
        scale_215 = torch.nn.functional.adaptive_avg_pool2d(input_478, 1)
        scale_216 = torch.conv2d(
            scale_215,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_215 = l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_217 = torch.nn.functional.silu(scale_216, inplace=True)
        scale_216 = None
        scale_218 = torch.conv2d(
            scale_217,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_217 = l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_219 = torch.sigmoid(scale_218)
        scale_218 = None
        input_479 = scale_219 * input_478
        scale_219 = input_478 = None
        input_480 = torch.conv2d(
            input_479,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_479 = l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_481 = torch.nn.functional.batch_norm(
            input_480,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_480 = l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_56 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_56 = None
        input_481 += result_55
        result_56 = input_481
        input_481 = result_55 = None
        input_482 = torch.conv2d(
            result_56,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_483 = torch.nn.functional.batch_norm(
            input_482,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_482 = l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_484 = torch.nn.functional.silu(input_483, inplace=True)
        input_483 = None
        input_485 = torch.conv2d(
            input_484,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_484 = l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_486 = torch.nn.functional.batch_norm(
            input_485,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_485 = l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_487 = torch.nn.functional.silu(input_486, inplace=True)
        input_486 = None
        scale_220 = torch.nn.functional.adaptive_avg_pool2d(input_487, 1)
        scale_221 = torch.conv2d(
            scale_220,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_220 = l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_222 = torch.nn.functional.silu(scale_221, inplace=True)
        scale_221 = None
        scale_223 = torch.conv2d(
            scale_222,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_222 = l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_224 = torch.sigmoid(scale_223)
        scale_223 = None
        input_488 = scale_224 * input_487
        scale_224 = input_487 = None
        input_489 = torch.conv2d(
            input_488,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_488 = l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_490 = torch.nn.functional.batch_norm(
            input_489,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_489 = l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_57 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_57 = None
        input_490 += result_56
        result_57 = input_490
        input_490 = result_56 = None
        input_491 = torch.conv2d(
            result_57,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_492 = torch.nn.functional.batch_norm(
            input_491,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_491 = l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_493 = torch.nn.functional.silu(input_492, inplace=True)
        input_492 = None
        input_494 = torch.conv2d(
            input_493,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_493 = l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_495 = torch.nn.functional.batch_norm(
            input_494,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_494 = l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_496 = torch.nn.functional.silu(input_495, inplace=True)
        input_495 = None
        scale_225 = torch.nn.functional.adaptive_avg_pool2d(input_496, 1)
        scale_226 = torch.conv2d(
            scale_225,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_225 = l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_227 = torch.nn.functional.silu(scale_226, inplace=True)
        scale_226 = None
        scale_228 = torch.conv2d(
            scale_227,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_227 = l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_229 = torch.sigmoid(scale_228)
        scale_228 = None
        input_497 = scale_229 * input_496
        scale_229 = input_496 = None
        input_498 = torch.conv2d(
            input_497,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_497 = l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_499 = torch.nn.functional.batch_norm(
            input_498,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_498 = l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_58 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_58 = None
        input_499 += result_57
        result_58 = input_499
        input_499 = result_57 = None
        input_500 = torch.conv2d(
            result_58,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_501 = torch.nn.functional.batch_norm(
            input_500,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_500 = l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_502 = torch.nn.functional.silu(input_501, inplace=True)
        input_501 = None
        input_503 = torch.conv2d(
            input_502,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_502 = l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_504 = torch.nn.functional.batch_norm(
            input_503,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_503 = l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_505 = torch.nn.functional.silu(input_504, inplace=True)
        input_504 = None
        scale_230 = torch.nn.functional.adaptive_avg_pool2d(input_505, 1)
        scale_231 = torch.conv2d(
            scale_230,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_230 = l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_232 = torch.nn.functional.silu(scale_231, inplace=True)
        scale_231 = None
        scale_233 = torch.conv2d(
            scale_232,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_232 = l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_234 = torch.sigmoid(scale_233)
        scale_233 = None
        input_506 = scale_234 * input_505
        scale_234 = input_505 = None
        input_507 = torch.conv2d(
            input_506,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_506 = l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_508 = torch.nn.functional.batch_norm(
            input_507,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_507 = l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_59 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_59 = None
        input_508 += result_58
        result_59 = input_508
        input_508 = result_58 = None
        input_509 = torch.conv2d(
            result_59,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_510 = torch.nn.functional.batch_norm(
            input_509,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_509 = l_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_18_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_511 = torch.nn.functional.silu(input_510, inplace=True)
        input_510 = None
        input_512 = torch.conv2d(
            input_511,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_511 = l_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_513 = torch.nn.functional.batch_norm(
            input_512,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_512 = l_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_18_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_514 = torch.nn.functional.silu(input_513, inplace=True)
        input_513 = None
        scale_235 = torch.nn.functional.adaptive_avg_pool2d(input_514, 1)
        scale_236 = torch.conv2d(
            scale_235,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_235 = l_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_237 = torch.nn.functional.silu(scale_236, inplace=True)
        scale_236 = None
        scale_238 = torch.conv2d(
            scale_237,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_237 = l_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_18_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_239 = torch.sigmoid(scale_238)
        scale_238 = None
        input_515 = scale_239 * input_514
        scale_239 = input_514 = None
        input_516 = torch.conv2d(
            input_515,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_515 = l_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_517 = torch.nn.functional.batch_norm(
            input_516,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_516 = l_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_18_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_60 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_60 = None
        input_517 += result_59
        result_60 = input_517
        input_517 = result_59 = None
        input_518 = torch.conv2d(
            result_60,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_519 = torch.nn.functional.batch_norm(
            input_518,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_518 = l_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_19_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_520 = torch.nn.functional.silu(input_519, inplace=True)
        input_519 = None
        input_521 = torch.conv2d(
            input_520,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_520 = l_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_522 = torch.nn.functional.batch_norm(
            input_521,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_521 = l_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_19_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_523 = torch.nn.functional.silu(input_522, inplace=True)
        input_522 = None
        scale_240 = torch.nn.functional.adaptive_avg_pool2d(input_523, 1)
        scale_241 = torch.conv2d(
            scale_240,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_240 = l_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_242 = torch.nn.functional.silu(scale_241, inplace=True)
        scale_241 = None
        scale_243 = torch.conv2d(
            scale_242,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_242 = l_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_19_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_244 = torch.sigmoid(scale_243)
        scale_243 = None
        input_524 = scale_244 * input_523
        scale_244 = input_523 = None
        input_525 = torch.conv2d(
            input_524,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_524 = l_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_526 = torch.nn.functional.batch_norm(
            input_525,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_525 = l_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_19_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_61 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_61 = None
        input_526 += result_60
        result_61 = input_526
        input_526 = result_60 = None
        input_527 = torch.conv2d(
            result_61,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_528 = torch.nn.functional.batch_norm(
            input_527,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_527 = l_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_20_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_529 = torch.nn.functional.silu(input_528, inplace=True)
        input_528 = None
        input_530 = torch.conv2d(
            input_529,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_529 = l_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_531 = torch.nn.functional.batch_norm(
            input_530,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_530 = l_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_20_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_532 = torch.nn.functional.silu(input_531, inplace=True)
        input_531 = None
        scale_245 = torch.nn.functional.adaptive_avg_pool2d(input_532, 1)
        scale_246 = torch.conv2d(
            scale_245,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_245 = l_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_247 = torch.nn.functional.silu(scale_246, inplace=True)
        scale_246 = None
        scale_248 = torch.conv2d(
            scale_247,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_247 = l_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_20_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_249 = torch.sigmoid(scale_248)
        scale_248 = None
        input_533 = scale_249 * input_532
        scale_249 = input_532 = None
        input_534 = torch.conv2d(
            input_533,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_533 = l_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_535 = torch.nn.functional.batch_norm(
            input_534,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_534 = l_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_20_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_62 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_62 = None
        input_535 += result_61
        result_62 = input_535
        input_535 = result_61 = None
        input_536 = torch.conv2d(
            result_62,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_537 = torch.nn.functional.batch_norm(
            input_536,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_536 = l_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_21_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_538 = torch.nn.functional.silu(input_537, inplace=True)
        input_537 = None
        input_539 = torch.conv2d(
            input_538,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_538 = l_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_540 = torch.nn.functional.batch_norm(
            input_539,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_539 = l_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_21_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_541 = torch.nn.functional.silu(input_540, inplace=True)
        input_540 = None
        scale_250 = torch.nn.functional.adaptive_avg_pool2d(input_541, 1)
        scale_251 = torch.conv2d(
            scale_250,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_250 = l_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_252 = torch.nn.functional.silu(scale_251, inplace=True)
        scale_251 = None
        scale_253 = torch.conv2d(
            scale_252,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_252 = l_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_21_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_254 = torch.sigmoid(scale_253)
        scale_253 = None
        input_542 = scale_254 * input_541
        scale_254 = input_541 = None
        input_543 = torch.conv2d(
            input_542,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_542 = l_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_544 = torch.nn.functional.batch_norm(
            input_543,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_543 = l_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_21_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_63 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_63 = None
        input_544 += result_62
        result_63 = input_544
        input_544 = result_62 = None
        input_545 = torch.conv2d(
            result_63,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_546 = torch.nn.functional.batch_norm(
            input_545,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_545 = l_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_22_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_547 = torch.nn.functional.silu(input_546, inplace=True)
        input_546 = None
        input_548 = torch.conv2d(
            input_547,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_547 = l_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_549 = torch.nn.functional.batch_norm(
            input_548,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_548 = l_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_22_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_550 = torch.nn.functional.silu(input_549, inplace=True)
        input_549 = None
        scale_255 = torch.nn.functional.adaptive_avg_pool2d(input_550, 1)
        scale_256 = torch.conv2d(
            scale_255,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_255 = l_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_257 = torch.nn.functional.silu(scale_256, inplace=True)
        scale_256 = None
        scale_258 = torch.conv2d(
            scale_257,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_257 = l_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_22_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_259 = torch.sigmoid(scale_258)
        scale_258 = None
        input_551 = scale_259 * input_550
        scale_259 = input_550 = None
        input_552 = torch.conv2d(
            input_551,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_551 = l_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_553 = torch.nn.functional.batch_norm(
            input_552,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_552 = l_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_22_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_64 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_64 = None
        input_553 += result_63
        result_64 = input_553
        input_553 = result_63 = None
        input_554 = torch.conv2d(
            result_64,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_555 = torch.nn.functional.batch_norm(
            input_554,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_554 = l_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_23_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_556 = torch.nn.functional.silu(input_555, inplace=True)
        input_555 = None
        input_557 = torch.conv2d(
            input_556,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_556 = l_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_558 = torch.nn.functional.batch_norm(
            input_557,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_557 = l_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_23_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_559 = torch.nn.functional.silu(input_558, inplace=True)
        input_558 = None
        scale_260 = torch.nn.functional.adaptive_avg_pool2d(input_559, 1)
        scale_261 = torch.conv2d(
            scale_260,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_260 = l_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_262 = torch.nn.functional.silu(scale_261, inplace=True)
        scale_261 = None
        scale_263 = torch.conv2d(
            scale_262,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_262 = l_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_23_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_264 = torch.sigmoid(scale_263)
        scale_263 = None
        input_560 = scale_264 * input_559
        scale_264 = input_559 = None
        input_561 = torch.conv2d(
            input_560,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_560 = l_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_562 = torch.nn.functional.batch_norm(
            input_561,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_561 = l_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_23_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_65 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_65 = None
        input_562 += result_64
        result_65 = input_562
        input_562 = result_64 = None
        input_563 = torch.conv2d(
            result_65,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_564 = torch.nn.functional.batch_norm(
            input_563,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_563 = l_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_24_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_565 = torch.nn.functional.silu(input_564, inplace=True)
        input_564 = None
        input_566 = torch.conv2d(
            input_565,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_565 = l_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_567 = torch.nn.functional.batch_norm(
            input_566,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_566 = l_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_24_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_568 = torch.nn.functional.silu(input_567, inplace=True)
        input_567 = None
        scale_265 = torch.nn.functional.adaptive_avg_pool2d(input_568, 1)
        scale_266 = torch.conv2d(
            scale_265,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_265 = l_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_267 = torch.nn.functional.silu(scale_266, inplace=True)
        scale_266 = None
        scale_268 = torch.conv2d(
            scale_267,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_267 = l_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_24_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_269 = torch.sigmoid(scale_268)
        scale_268 = None
        input_569 = scale_269 * input_568
        scale_269 = input_568 = None
        input_570 = torch.conv2d(
            input_569,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_569 = l_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_571 = torch.nn.functional.batch_norm(
            input_570,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_570 = l_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_24_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_66 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_66 = None
        input_571 += result_65
        result_66 = input_571
        input_571 = result_65 = None
        input_572 = torch.conv2d(
            result_66,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_66 = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_573 = torch.nn.functional.batch_norm(
            input_572,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_572 = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_574 = torch.nn.functional.silu(input_573, inplace=True)
        input_573 = None
        input_575 = torch.conv2d(
            input_574,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_574 = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_576 = torch.nn.functional.batch_norm(
            input_575,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_575 = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_577 = torch.nn.functional.silu(input_576, inplace=True)
        input_576 = None
        scale_270 = torch.nn.functional.adaptive_avg_pool2d(input_577, 1)
        scale_271 = torch.conv2d(
            scale_270,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_270 = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_272 = torch.nn.functional.silu(scale_271, inplace=True)
        scale_271 = None
        scale_273 = torch.conv2d(
            scale_272,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_272 = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_274 = torch.sigmoid(scale_273)
        scale_273 = None
        input_578 = scale_274 * input_577
        scale_274 = input_577 = None
        input_579 = torch.conv2d(
            input_578,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_578 = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_580 = torch.nn.functional.batch_norm(
            input_579,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_579 = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_581 = torch.conv2d(
            input_580,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_582 = torch.nn.functional.batch_norm(
            input_581,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_581 = l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_583 = torch.nn.functional.silu(input_582, inplace=True)
        input_582 = None
        input_584 = torch.conv2d(
            input_583,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3840,
        )
        input_583 = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_585 = torch.nn.functional.batch_norm(
            input_584,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_584 = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_586 = torch.nn.functional.silu(input_585, inplace=True)
        input_585 = None
        scale_275 = torch.nn.functional.adaptive_avg_pool2d(input_586, 1)
        scale_276 = torch.conv2d(
            scale_275,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_275 = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_277 = torch.nn.functional.silu(scale_276, inplace=True)
        scale_276 = None
        scale_278 = torch.conv2d(
            scale_277,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_277 = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_279 = torch.sigmoid(scale_278)
        scale_278 = None
        input_587 = scale_279 * input_586
        scale_279 = input_586 = None
        input_588 = torch.conv2d(
            input_587,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_587 = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_589 = torch.nn.functional.batch_norm(
            input_588,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_588 = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_67 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_67 = None
        input_589 += input_580
        result_67 = input_589
        input_589 = input_580 = None
        input_590 = torch.conv2d(
            result_67,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_591 = torch.nn.functional.batch_norm(
            input_590,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_590 = l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_592 = torch.nn.functional.silu(input_591, inplace=True)
        input_591 = None
        input_593 = torch.conv2d(
            input_592,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3840,
        )
        input_592 = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_594 = torch.nn.functional.batch_norm(
            input_593,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_593 = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_595 = torch.nn.functional.silu(input_594, inplace=True)
        input_594 = None
        scale_280 = torch.nn.functional.adaptive_avg_pool2d(input_595, 1)
        scale_281 = torch.conv2d(
            scale_280,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_280 = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_282 = torch.nn.functional.silu(scale_281, inplace=True)
        scale_281 = None
        scale_283 = torch.conv2d(
            scale_282,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_282 = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_284 = torch.sigmoid(scale_283)
        scale_283 = None
        input_596 = scale_284 * input_595
        scale_284 = input_595 = None
        input_597 = torch.conv2d(
            input_596,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_596 = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_598 = torch.nn.functional.batch_norm(
            input_597,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_597 = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_68 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_68 = None
        input_598 += result_67
        result_68 = input_598
        input_598 = result_67 = None
        input_599 = torch.conv2d(
            result_68,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_600 = torch.nn.functional.batch_norm(
            input_599,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_599 = l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_601 = torch.nn.functional.silu(input_600, inplace=True)
        input_600 = None
        input_602 = torch.conv2d(
            input_601,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3840,
        )
        input_601 = l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_603 = torch.nn.functional.batch_norm(
            input_602,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_602 = l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_604 = torch.nn.functional.silu(input_603, inplace=True)
        input_603 = None
        scale_285 = torch.nn.functional.adaptive_avg_pool2d(input_604, 1)
        scale_286 = torch.conv2d(
            scale_285,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_285 = l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_287 = torch.nn.functional.silu(scale_286, inplace=True)
        scale_286 = None
        scale_288 = torch.conv2d(
            scale_287,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_287 = l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_289 = torch.sigmoid(scale_288)
        scale_288 = None
        input_605 = scale_289 * input_604
        scale_289 = input_604 = None
        input_606 = torch.conv2d(
            input_605,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_605 = l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_607 = torch.nn.functional.batch_norm(
            input_606,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_606 = l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_69 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_69 = None
        input_607 += result_68
        result_69 = input_607
        input_607 = result_68 = None
        input_608 = torch.conv2d(
            result_69,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_609 = torch.nn.functional.batch_norm(
            input_608,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_608 = l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_610 = torch.nn.functional.silu(input_609, inplace=True)
        input_609 = None
        input_611 = torch.conv2d(
            input_610,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3840,
        )
        input_610 = l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_612 = torch.nn.functional.batch_norm(
            input_611,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_611 = l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_613 = torch.nn.functional.silu(input_612, inplace=True)
        input_612 = None
        scale_290 = torch.nn.functional.adaptive_avg_pool2d(input_613, 1)
        scale_291 = torch.conv2d(
            scale_290,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_290 = l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_292 = torch.nn.functional.silu(scale_291, inplace=True)
        scale_291 = None
        scale_293 = torch.conv2d(
            scale_292,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_292 = l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_294 = torch.sigmoid(scale_293)
        scale_293 = None
        input_614 = scale_294 * input_613
        scale_294 = input_613 = None
        input_615 = torch.conv2d(
            input_614,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_614 = l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_616 = torch.nn.functional.batch_norm(
            input_615,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_615 = l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_70 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_70 = None
        input_616 += result_69
        result_70 = input_616
        input_616 = result_69 = None
        input_617 = torch.conv2d(
            result_70,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_618 = torch.nn.functional.batch_norm(
            input_617,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_617 = l_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_619 = torch.nn.functional.silu(input_618, inplace=True)
        input_618 = None
        input_620 = torch.conv2d(
            input_619,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3840,
        )
        input_619 = l_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_621 = torch.nn.functional.batch_norm(
            input_620,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_620 = l_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_622 = torch.nn.functional.silu(input_621, inplace=True)
        input_621 = None
        scale_295 = torch.nn.functional.adaptive_avg_pool2d(input_622, 1)
        scale_296 = torch.conv2d(
            scale_295,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_295 = l_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_297 = torch.nn.functional.silu(scale_296, inplace=True)
        scale_296 = None
        scale_298 = torch.conv2d(
            scale_297,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_297 = l_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_299 = torch.sigmoid(scale_298)
        scale_298 = None
        input_623 = scale_299 * input_622
        scale_299 = input_622 = None
        input_624 = torch.conv2d(
            input_623,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_623 = l_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_625 = torch.nn.functional.batch_norm(
            input_624,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_624 = l_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_71 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_71 = None
        input_625 += result_70
        result_71 = input_625
        input_625 = result_70 = None
        input_626 = torch.conv2d(
            result_71,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_627 = torch.nn.functional.batch_norm(
            input_626,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_626 = l_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_628 = torch.nn.functional.silu(input_627, inplace=True)
        input_627 = None
        input_629 = torch.conv2d(
            input_628,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3840,
        )
        input_628 = l_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_630 = torch.nn.functional.batch_norm(
            input_629,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_629 = l_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_631 = torch.nn.functional.silu(input_630, inplace=True)
        input_630 = None
        scale_300 = torch.nn.functional.adaptive_avg_pool2d(input_631, 1)
        scale_301 = torch.conv2d(
            scale_300,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_300 = l_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_302 = torch.nn.functional.silu(scale_301, inplace=True)
        scale_301 = None
        scale_303 = torch.conv2d(
            scale_302,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_302 = l_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_304 = torch.sigmoid(scale_303)
        scale_303 = None
        input_632 = scale_304 * input_631
        scale_304 = input_631 = None
        input_633 = torch.conv2d(
            input_632,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_632 = l_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_634 = torch.nn.functional.batch_norm(
            input_633,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_633 = l_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_72 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_72 = None
        input_634 += result_71
        result_72 = input_634
        input_634 = result_71 = None
        input_635 = torch.conv2d(
            result_72,
            l_self_modules_features_modules_8_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_72 = (
            l_self_modules_features_modules_8_modules_0_parameters_weight_
        ) = None
        input_636 = torch.nn.functional.batch_norm(
            input_635,
            l_self_modules_features_modules_8_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_1_buffers_running_var_,
            l_self_modules_features_modules_8_modules_1_parameters_weight_,
            l_self_modules_features_modules_8_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_635 = (
            l_self_modules_features_modules_8_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_8_modules_1_buffers_running_var_
        ) = (
            l_self_modules_features_modules_8_modules_1_parameters_weight_
        ) = l_self_modules_features_modules_8_modules_1_parameters_bias_ = None
        input_637 = torch.nn.functional.silu(input_636, inplace=True)
        input_636 = None
        x = torch.nn.functional.adaptive_avg_pool2d(input_637, 1)
        input_637 = None
        x_1 = torch.flatten(x, 1)
        x = None
        input_638 = torch.nn.functional.dropout(x_1, 0.4, False, True)
        x_1 = None
        input_639 = torch._C._nn.linear(
            input_638,
            l_self_modules_classifier_modules_1_parameters_weight_,
            l_self_modules_classifier_modules_1_parameters_bias_,
        )
        input_638 = (
            l_self_modules_classifier_modules_1_parameters_weight_
        ) = l_self_modules_classifier_modules_1_parameters_bias_ = None
        return (input_639,)
