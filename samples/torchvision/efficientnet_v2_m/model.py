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
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        result_2 = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_14 = torch.nn.functional.batch_norm(
            input_13,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_13 = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_15 = torch.nn.functional.silu(input_14, inplace=True)
        input_14 = None
        input_16 = torch.conv2d(
            input_15,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_15 = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_17 = torch.nn.functional.batch_norm(
            input_16,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_16 = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_18 = torch.conv2d(
            input_17,
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
        input_19 = torch.nn.functional.batch_norm(
            input_18,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_18 = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_20 = torch.nn.functional.silu(input_19, inplace=True)
        input_19 = None
        input_21 = torch.conv2d(
            input_20,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_20 = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_22 = torch.nn.functional.batch_norm(
            input_21,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_21 = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_3 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_3 = None
        input_22 += input_17
        result_3 = input_22
        input_22 = input_17 = None
        input_23 = torch.conv2d(
            result_3,
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
        input_24 = torch.nn.functional.batch_norm(
            input_23,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_23 = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_25 = torch.nn.functional.silu(input_24, inplace=True)
        input_24 = None
        input_26 = torch.conv2d(
            input_25,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_25 = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_27 = torch.nn.functional.batch_norm(
            input_26,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_26 = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_4 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_4 = None
        input_27 += result_3
        result_4 = input_27
        input_27 = result_3 = None
        input_28 = torch.conv2d(
            result_4,
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
        input_29 = torch.nn.functional.batch_norm(
            input_28,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_28 = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_30 = torch.nn.functional.silu(input_29, inplace=True)
        input_29 = None
        input_31 = torch.conv2d(
            input_30,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_30 = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_32 = torch.nn.functional.batch_norm(
            input_31,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_31 = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_5 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_5 = None
        input_32 += result_4
        result_5 = input_32
        input_32 = result_4 = None
        input_33 = torch.conv2d(
            result_5,
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
        input_34 = torch.nn.functional.batch_norm(
            input_33,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_33 = l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_35 = torch.nn.functional.silu(input_34, inplace=True)
        input_34 = None
        input_36 = torch.conv2d(
            input_35,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_35 = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_37 = torch.nn.functional.batch_norm(
            input_36,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_36 = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_6 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_6 = None
        input_37 += result_5
        result_6 = input_37
        input_37 = result_5 = None
        input_38 = torch.conv2d(
            result_6,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        result_6 = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_39 = torch.nn.functional.batch_norm(
            input_38,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_38 = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_40 = torch.nn.functional.silu(input_39, inplace=True)
        input_39 = None
        input_41 = torch.conv2d(
            input_40,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_40 = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_42 = torch.nn.functional.batch_norm(
            input_41,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_41 = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_43 = torch.conv2d(
            input_42,
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
        input_44 = torch.nn.functional.batch_norm(
            input_43,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_43 = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_45 = torch.nn.functional.silu(input_44, inplace=True)
        input_44 = None
        input_46 = torch.conv2d(
            input_45,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_45 = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_47 = torch.nn.functional.batch_norm(
            input_46,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_46 = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_7 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_7 = None
        input_47 += input_42
        result_7 = input_47
        input_47 = input_42 = None
        input_48 = torch.conv2d(
            result_7,
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
        input_49 = torch.nn.functional.batch_norm(
            input_48,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_48 = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_50 = torch.nn.functional.silu(input_49, inplace=True)
        input_49 = None
        input_51 = torch.conv2d(
            input_50,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_50 = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_52 = torch.nn.functional.batch_norm(
            input_51,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_51 = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_8 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_8 = None
        input_52 += result_7
        result_8 = input_52
        input_52 = result_7 = None
        input_53 = torch.conv2d(
            result_8,
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
        input_54 = torch.nn.functional.batch_norm(
            input_53,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_53 = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_55 = torch.nn.functional.silu(input_54, inplace=True)
        input_54 = None
        input_56 = torch.conv2d(
            input_55,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_55 = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_57 = torch.nn.functional.batch_norm(
            input_56,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_56 = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_9 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_9 = None
        input_57 += result_8
        result_9 = input_57
        input_57 = result_8 = None
        input_58 = torch.conv2d(
            result_9,
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
        input_59 = torch.nn.functional.batch_norm(
            input_58,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_58 = l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_60 = torch.nn.functional.silu(input_59, inplace=True)
        input_59 = None
        input_61 = torch.conv2d(
            input_60,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_60 = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_62 = torch.nn.functional.batch_norm(
            input_61,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_61 = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_10 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_10 = None
        input_62 += result_9
        result_10 = input_62
        input_62 = result_9 = None
        input_63 = torch.conv2d(
            result_10,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_10 = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_64 = torch.nn.functional.batch_norm(
            input_63,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_63 = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_65 = torch.nn.functional.silu(input_64, inplace=True)
        input_64 = None
        input_66 = torch.conv2d(
            input_65,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            320,
        )
        input_65 = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_67 = torch.nn.functional.batch_norm(
            input_66,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_66 = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_68 = torch.nn.functional.silu(input_67, inplace=True)
        input_67 = None
        scale = torch.nn.functional.adaptive_avg_pool2d(input_68, 1)
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
        input_69 = scale_4 * input_68
        scale_4 = input_68 = None
        input_70 = torch.conv2d(
            input_69,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_69 = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_71 = torch.nn.functional.batch_norm(
            input_70,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_70 = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_72 = torch.conv2d(
            input_71,
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
        input_73 = torch.nn.functional.batch_norm(
            input_72,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_72 = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_74 = torch.nn.functional.silu(input_73, inplace=True)
        input_73 = None
        input_75 = torch.conv2d(
            input_74,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            640,
        )
        input_74 = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_76 = torch.nn.functional.batch_norm(
            input_75,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_75 = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_77 = torch.nn.functional.silu(input_76, inplace=True)
        input_76 = None
        scale_5 = torch.nn.functional.adaptive_avg_pool2d(input_77, 1)
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
        input_78 = scale_9 * input_77
        scale_9 = input_77 = None
        input_79 = torch.conv2d(
            input_78,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_78 = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_80 = torch.nn.functional.batch_norm(
            input_79,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_79 = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_11 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_11 = None
        input_80 += input_71
        result_11 = input_80
        input_80 = input_71 = None
        input_81 = torch.conv2d(
            result_11,
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
        input_82 = torch.nn.functional.batch_norm(
            input_81,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_81 = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_83 = torch.nn.functional.silu(input_82, inplace=True)
        input_82 = None
        input_84 = torch.conv2d(
            input_83,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            640,
        )
        input_83 = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_85 = torch.nn.functional.batch_norm(
            input_84,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_84 = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_86 = torch.nn.functional.silu(input_85, inplace=True)
        input_85 = None
        scale_10 = torch.nn.functional.adaptive_avg_pool2d(input_86, 1)
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
        input_87 = scale_14 * input_86
        scale_14 = input_86 = None
        input_88 = torch.conv2d(
            input_87,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_87 = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_89 = torch.nn.functional.batch_norm(
            input_88,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_88 = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_12 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_12 = None
        input_89 += result_11
        result_12 = input_89
        input_89 = result_11 = None
        input_90 = torch.conv2d(
            result_12,
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
        input_91 = torch.nn.functional.batch_norm(
            input_90,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_90 = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_92 = torch.nn.functional.silu(input_91, inplace=True)
        input_91 = None
        input_93 = torch.conv2d(
            input_92,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            640,
        )
        input_92 = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_94 = torch.nn.functional.batch_norm(
            input_93,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_93 = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_95 = torch.nn.functional.silu(input_94, inplace=True)
        input_94 = None
        scale_15 = torch.nn.functional.adaptive_avg_pool2d(input_95, 1)
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
        input_96 = scale_19 * input_95
        scale_19 = input_95 = None
        input_97 = torch.conv2d(
            input_96,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_96 = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_98 = torch.nn.functional.batch_norm(
            input_97,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_97 = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_13 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_13 = None
        input_98 += result_12
        result_13 = input_98
        input_98 = result_12 = None
        input_99 = torch.conv2d(
            result_13,
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
        input_100 = torch.nn.functional.batch_norm(
            input_99,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_99 = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_101 = torch.nn.functional.silu(input_100, inplace=True)
        input_100 = None
        input_102 = torch.conv2d(
            input_101,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            640,
        )
        input_101 = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_103 = torch.nn.functional.batch_norm(
            input_102,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_102 = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_104 = torch.nn.functional.silu(input_103, inplace=True)
        input_103 = None
        scale_20 = torch.nn.functional.adaptive_avg_pool2d(input_104, 1)
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
        input_105 = scale_24 * input_104
        scale_24 = input_104 = None
        input_106 = torch.conv2d(
            input_105,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_105 = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_107 = torch.nn.functional.batch_norm(
            input_106,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_106 = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_14 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_14 = None
        input_107 += result_13
        result_14 = input_107
        input_107 = result_13 = None
        input_108 = torch.conv2d(
            result_14,
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
        input_109 = torch.nn.functional.batch_norm(
            input_108,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_108 = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_110 = torch.nn.functional.silu(input_109, inplace=True)
        input_109 = None
        input_111 = torch.conv2d(
            input_110,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            640,
        )
        input_110 = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_112 = torch.nn.functional.batch_norm(
            input_111,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_111 = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_113 = torch.nn.functional.silu(input_112, inplace=True)
        input_112 = None
        scale_25 = torch.nn.functional.adaptive_avg_pool2d(input_113, 1)
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
        input_114 = scale_29 * input_113
        scale_29 = input_113 = None
        input_115 = torch.conv2d(
            input_114,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_114 = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_116 = torch.nn.functional.batch_norm(
            input_115,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_115 = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_15 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_15 = None
        input_116 += result_14
        result_15 = input_116
        input_116 = result_14 = None
        input_117 = torch.conv2d(
            result_15,
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
        input_118 = torch.nn.functional.batch_norm(
            input_117,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_117 = l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_119 = torch.nn.functional.silu(input_118, inplace=True)
        input_118 = None
        input_120 = torch.conv2d(
            input_119,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            640,
        )
        input_119 = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_121 = torch.nn.functional.batch_norm(
            input_120,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_120 = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_122 = torch.nn.functional.silu(input_121, inplace=True)
        input_121 = None
        scale_30 = torch.nn.functional.adaptive_avg_pool2d(input_122, 1)
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
        input_123 = scale_34 * input_122
        scale_34 = input_122 = None
        input_124 = torch.conv2d(
            input_123,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_123 = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_125 = torch.nn.functional.batch_norm(
            input_124,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_124 = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_16 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_16 = None
        input_125 += result_15
        result_16 = input_125
        input_125 = result_15 = None
        input_126 = torch.conv2d(
            result_16,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_16 = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_127 = torch.nn.functional.batch_norm(
            input_126,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_126 = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_128 = torch.nn.functional.silu(input_127, inplace=True)
        input_127 = None
        input_129 = torch.conv2d(
            input_128,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_128 = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_130 = torch.nn.functional.batch_norm(
            input_129,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_129 = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_131 = torch.nn.functional.silu(input_130, inplace=True)
        input_130 = None
        scale_35 = torch.nn.functional.adaptive_avg_pool2d(input_131, 1)
        scale_36 = torch.conv2d(
            scale_35,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_35 = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_37 = torch.nn.functional.silu(scale_36, inplace=True)
        scale_36 = None
        scale_38 = torch.conv2d(
            scale_37,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_37 = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_39 = torch.sigmoid(scale_38)
        scale_38 = None
        input_132 = scale_39 * input_131
        scale_39 = input_131 = None
        input_133 = torch.conv2d(
            input_132,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_132 = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_134 = torch.nn.functional.batch_norm(
            input_133,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_133 = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_135 = torch.conv2d(
            input_134,
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
        input_136 = torch.nn.functional.batch_norm(
            input_135,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_135 = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_137 = torch.nn.functional.silu(input_136, inplace=True)
        input_136 = None
        input_138 = torch.conv2d(
            input_137,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1056,
        )
        input_137 = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_139 = torch.nn.functional.batch_norm(
            input_138,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_138 = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_140 = torch.nn.functional.silu(input_139, inplace=True)
        input_139 = None
        scale_40 = torch.nn.functional.adaptive_avg_pool2d(input_140, 1)
        scale_41 = torch.conv2d(
            scale_40,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_40 = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_42 = torch.nn.functional.silu(scale_41, inplace=True)
        scale_41 = None
        scale_43 = torch.conv2d(
            scale_42,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_42 = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_44 = torch.sigmoid(scale_43)
        scale_43 = None
        input_141 = scale_44 * input_140
        scale_44 = input_140 = None
        input_142 = torch.conv2d(
            input_141,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_141 = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_143 = torch.nn.functional.batch_norm(
            input_142,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_142 = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_17 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_17 = None
        input_143 += input_134
        result_17 = input_143
        input_143 = input_134 = None
        input_144 = torch.conv2d(
            result_17,
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
        input_145 = torch.nn.functional.batch_norm(
            input_144,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_144 = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_146 = torch.nn.functional.silu(input_145, inplace=True)
        input_145 = None
        input_147 = torch.conv2d(
            input_146,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1056,
        )
        input_146 = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_148 = torch.nn.functional.batch_norm(
            input_147,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_147 = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_149 = torch.nn.functional.silu(input_148, inplace=True)
        input_148 = None
        scale_45 = torch.nn.functional.adaptive_avg_pool2d(input_149, 1)
        scale_46 = torch.conv2d(
            scale_45,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_45 = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_47 = torch.nn.functional.silu(scale_46, inplace=True)
        scale_46 = None
        scale_48 = torch.conv2d(
            scale_47,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_47 = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_49 = torch.sigmoid(scale_48)
        scale_48 = None
        input_150 = scale_49 * input_149
        scale_49 = input_149 = None
        input_151 = torch.conv2d(
            input_150,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_150 = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_152 = torch.nn.functional.batch_norm(
            input_151,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_151 = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_18 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_18 = None
        input_152 += result_17
        result_18 = input_152
        input_152 = result_17 = None
        input_153 = torch.conv2d(
            result_18,
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
        input_154 = torch.nn.functional.batch_norm(
            input_153,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_153 = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_155 = torch.nn.functional.silu(input_154, inplace=True)
        input_154 = None
        input_156 = torch.conv2d(
            input_155,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1056,
        )
        input_155 = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_157 = torch.nn.functional.batch_norm(
            input_156,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_156 = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_158 = torch.nn.functional.silu(input_157, inplace=True)
        input_157 = None
        scale_50 = torch.nn.functional.adaptive_avg_pool2d(input_158, 1)
        scale_51 = torch.conv2d(
            scale_50,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_50 = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_52 = torch.nn.functional.silu(scale_51, inplace=True)
        scale_51 = None
        scale_53 = torch.conv2d(
            scale_52,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_52 = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_54 = torch.sigmoid(scale_53)
        scale_53 = None
        input_159 = scale_54 * input_158
        scale_54 = input_158 = None
        input_160 = torch.conv2d(
            input_159,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_159 = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_161 = torch.nn.functional.batch_norm(
            input_160,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_160 = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_19 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_19 = None
        input_161 += result_18
        result_19 = input_161
        input_161 = result_18 = None
        input_162 = torch.conv2d(
            result_19,
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
        input_163 = torch.nn.functional.batch_norm(
            input_162,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_162 = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_164 = torch.nn.functional.silu(input_163, inplace=True)
        input_163 = None
        input_165 = torch.conv2d(
            input_164,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1056,
        )
        input_164 = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_166 = torch.nn.functional.batch_norm(
            input_165,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_165 = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_167 = torch.nn.functional.silu(input_166, inplace=True)
        input_166 = None
        scale_55 = torch.nn.functional.adaptive_avg_pool2d(input_167, 1)
        scale_56 = torch.conv2d(
            scale_55,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_55 = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_57 = torch.nn.functional.silu(scale_56, inplace=True)
        scale_56 = None
        scale_58 = torch.conv2d(
            scale_57,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_57 = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_59 = torch.sigmoid(scale_58)
        scale_58 = None
        input_168 = scale_59 * input_167
        scale_59 = input_167 = None
        input_169 = torch.conv2d(
            input_168,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_168 = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_170 = torch.nn.functional.batch_norm(
            input_169,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_169 = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_20 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_20 = None
        input_170 += result_19
        result_20 = input_170
        input_170 = result_19 = None
        input_171 = torch.conv2d(
            result_20,
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
        input_172 = torch.nn.functional.batch_norm(
            input_171,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_171 = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_173 = torch.nn.functional.silu(input_172, inplace=True)
        input_172 = None
        input_174 = torch.conv2d(
            input_173,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1056,
        )
        input_173 = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_175 = torch.nn.functional.batch_norm(
            input_174,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_174 = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_176 = torch.nn.functional.silu(input_175, inplace=True)
        input_175 = None
        scale_60 = torch.nn.functional.adaptive_avg_pool2d(input_176, 1)
        scale_61 = torch.conv2d(
            scale_60,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_60 = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_62 = torch.nn.functional.silu(scale_61, inplace=True)
        scale_61 = None
        scale_63 = torch.conv2d(
            scale_62,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_62 = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_64 = torch.sigmoid(scale_63)
        scale_63 = None
        input_177 = scale_64 * input_176
        scale_64 = input_176 = None
        input_178 = torch.conv2d(
            input_177,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_177 = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_179 = torch.nn.functional.batch_norm(
            input_178,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_178 = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_21 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_21 = None
        input_179 += result_20
        result_21 = input_179
        input_179 = result_20 = None
        input_180 = torch.conv2d(
            result_21,
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
        input_181 = torch.nn.functional.batch_norm(
            input_180,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_180 = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_182 = torch.nn.functional.silu(input_181, inplace=True)
        input_181 = None
        input_183 = torch.conv2d(
            input_182,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1056,
        )
        input_182 = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_184 = torch.nn.functional.batch_norm(
            input_183,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_183 = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_185 = torch.nn.functional.silu(input_184, inplace=True)
        input_184 = None
        scale_65 = torch.nn.functional.adaptive_avg_pool2d(input_185, 1)
        scale_66 = torch.conv2d(
            scale_65,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_65 = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_67 = torch.nn.functional.silu(scale_66, inplace=True)
        scale_66 = None
        scale_68 = torch.conv2d(
            scale_67,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_67 = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_69 = torch.sigmoid(scale_68)
        scale_68 = None
        input_186 = scale_69 * input_185
        scale_69 = input_185 = None
        input_187 = torch.conv2d(
            input_186,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_186 = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_188 = torch.nn.functional.batch_norm(
            input_187,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_187 = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_22 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_22 = None
        input_188 += result_21
        result_22 = input_188
        input_188 = result_21 = None
        input_189 = torch.conv2d(
            result_22,
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
        input_190 = torch.nn.functional.batch_norm(
            input_189,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_189 = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_191 = torch.nn.functional.silu(input_190, inplace=True)
        input_190 = None
        input_192 = torch.conv2d(
            input_191,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1056,
        )
        input_191 = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_193 = torch.nn.functional.batch_norm(
            input_192,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_192 = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_194 = torch.nn.functional.silu(input_193, inplace=True)
        input_193 = None
        scale_70 = torch.nn.functional.adaptive_avg_pool2d(input_194, 1)
        scale_71 = torch.conv2d(
            scale_70,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_70 = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_72 = torch.nn.functional.silu(scale_71, inplace=True)
        scale_71 = None
        scale_73 = torch.conv2d(
            scale_72,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_72 = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_74 = torch.sigmoid(scale_73)
        scale_73 = None
        input_195 = scale_74 * input_194
        scale_74 = input_194 = None
        input_196 = torch.conv2d(
            input_195,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_195 = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_197 = torch.nn.functional.batch_norm(
            input_196,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_196 = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_23 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_23 = None
        input_197 += result_22
        result_23 = input_197
        input_197 = result_22 = None
        input_198 = torch.conv2d(
            result_23,
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
        input_199 = torch.nn.functional.batch_norm(
            input_198,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_198 = l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_200 = torch.nn.functional.silu(input_199, inplace=True)
        input_199 = None
        input_201 = torch.conv2d(
            input_200,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1056,
        )
        input_200 = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_202 = torch.nn.functional.batch_norm(
            input_201,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_201 = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_203 = torch.nn.functional.silu(input_202, inplace=True)
        input_202 = None
        scale_75 = torch.nn.functional.adaptive_avg_pool2d(input_203, 1)
        scale_76 = torch.conv2d(
            scale_75,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_75 = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_77 = torch.nn.functional.silu(scale_76, inplace=True)
        scale_76 = None
        scale_78 = torch.conv2d(
            scale_77,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_77 = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_79 = torch.sigmoid(scale_78)
        scale_78 = None
        input_204 = scale_79 * input_203
        scale_79 = input_203 = None
        input_205 = torch.conv2d(
            input_204,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_204 = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_206 = torch.nn.functional.batch_norm(
            input_205,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_205 = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_24 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_24 = None
        input_206 += result_23
        result_24 = input_206
        input_206 = result_23 = None
        input_207 = torch.conv2d(
            result_24,
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
        input_208 = torch.nn.functional.batch_norm(
            input_207,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_207 = l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_209 = torch.nn.functional.silu(input_208, inplace=True)
        input_208 = None
        input_210 = torch.conv2d(
            input_209,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1056,
        )
        input_209 = l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_211 = torch.nn.functional.batch_norm(
            input_210,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_210 = l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_212 = torch.nn.functional.silu(input_211, inplace=True)
        input_211 = None
        scale_80 = torch.nn.functional.adaptive_avg_pool2d(input_212, 1)
        scale_81 = torch.conv2d(
            scale_80,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_80 = l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_82 = torch.nn.functional.silu(scale_81, inplace=True)
        scale_81 = None
        scale_83 = torch.conv2d(
            scale_82,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_82 = l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_84 = torch.sigmoid(scale_83)
        scale_83 = None
        input_213 = scale_84 * input_212
        scale_84 = input_212 = None
        input_214 = torch.conv2d(
            input_213,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_213 = l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_215 = torch.nn.functional.batch_norm(
            input_214,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_214 = l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_25 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_25 = None
        input_215 += result_24
        result_25 = input_215
        input_215 = result_24 = None
        input_216 = torch.conv2d(
            result_25,
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
        input_217 = torch.nn.functional.batch_norm(
            input_216,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_216 = l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_218 = torch.nn.functional.silu(input_217, inplace=True)
        input_217 = None
        input_219 = torch.conv2d(
            input_218,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1056,
        )
        input_218 = l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_220 = torch.nn.functional.batch_norm(
            input_219,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_219 = l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_221 = torch.nn.functional.silu(input_220, inplace=True)
        input_220 = None
        scale_85 = torch.nn.functional.adaptive_avg_pool2d(input_221, 1)
        scale_86 = torch.conv2d(
            scale_85,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_85 = l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_87 = torch.nn.functional.silu(scale_86, inplace=True)
        scale_86 = None
        scale_88 = torch.conv2d(
            scale_87,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_87 = l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_89 = torch.sigmoid(scale_88)
        scale_88 = None
        input_222 = scale_89 * input_221
        scale_89 = input_221 = None
        input_223 = torch.conv2d(
            input_222,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_222 = l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_224 = torch.nn.functional.batch_norm(
            input_223,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_223 = l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_10_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_26 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_26 = None
        input_224 += result_25
        result_26 = input_224
        input_224 = result_25 = None
        input_225 = torch.conv2d(
            result_26,
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
        input_226 = torch.nn.functional.batch_norm(
            input_225,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_225 = l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_227 = torch.nn.functional.silu(input_226, inplace=True)
        input_226 = None
        input_228 = torch.conv2d(
            input_227,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1056,
        )
        input_227 = l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_229 = torch.nn.functional.batch_norm(
            input_228,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_228 = l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_230 = torch.nn.functional.silu(input_229, inplace=True)
        input_229 = None
        scale_90 = torch.nn.functional.adaptive_avg_pool2d(input_230, 1)
        scale_91 = torch.conv2d(
            scale_90,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_90 = l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_92 = torch.nn.functional.silu(scale_91, inplace=True)
        scale_91 = None
        scale_93 = torch.conv2d(
            scale_92,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_92 = l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_94 = torch.sigmoid(scale_93)
        scale_93 = None
        input_231 = scale_94 * input_230
        scale_94 = input_230 = None
        input_232 = torch.conv2d(
            input_231,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_231 = l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_233 = torch.nn.functional.batch_norm(
            input_232,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_232 = l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_11_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_27 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_27 = None
        input_233 += result_26
        result_27 = input_233
        input_233 = result_26 = None
        input_234 = torch.conv2d(
            result_27,
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
        input_235 = torch.nn.functional.batch_norm(
            input_234,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_234 = l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_236 = torch.nn.functional.silu(input_235, inplace=True)
        input_235 = None
        input_237 = torch.conv2d(
            input_236,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1056,
        )
        input_236 = l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_238 = torch.nn.functional.batch_norm(
            input_237,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_237 = l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_239 = torch.nn.functional.silu(input_238, inplace=True)
        input_238 = None
        scale_95 = torch.nn.functional.adaptive_avg_pool2d(input_239, 1)
        scale_96 = torch.conv2d(
            scale_95,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_95 = l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_97 = torch.nn.functional.silu(scale_96, inplace=True)
        scale_96 = None
        scale_98 = torch.conv2d(
            scale_97,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_97 = l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_99 = torch.sigmoid(scale_98)
        scale_98 = None
        input_240 = scale_99 * input_239
        scale_99 = input_239 = None
        input_241 = torch.conv2d(
            input_240,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_240 = l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_242 = torch.nn.functional.batch_norm(
            input_241,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_241 = l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_12_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_28 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_28 = None
        input_242 += result_27
        result_28 = input_242
        input_242 = result_27 = None
        input_243 = torch.conv2d(
            result_28,
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
        input_244 = torch.nn.functional.batch_norm(
            input_243,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_243 = l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_245 = torch.nn.functional.silu(input_244, inplace=True)
        input_244 = None
        input_246 = torch.conv2d(
            input_245,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1056,
        )
        input_245 = l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_247 = torch.nn.functional.batch_norm(
            input_246,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_246 = l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_248 = torch.nn.functional.silu(input_247, inplace=True)
        input_247 = None
        scale_100 = torch.nn.functional.adaptive_avg_pool2d(input_248, 1)
        scale_101 = torch.conv2d(
            scale_100,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_100 = l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_102 = torch.nn.functional.silu(scale_101, inplace=True)
        scale_101 = None
        scale_103 = torch.conv2d(
            scale_102,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_102 = l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_104 = torch.sigmoid(scale_103)
        scale_103 = None
        input_249 = scale_104 * input_248
        scale_104 = input_248 = None
        input_250 = torch.conv2d(
            input_249,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_249 = l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_251 = torch.nn.functional.batch_norm(
            input_250,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_250 = l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_13_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_29 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_29 = None
        input_251 += result_28
        result_29 = input_251
        input_251 = result_28 = None
        input_252 = torch.conv2d(
            result_29,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_29 = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_253 = torch.nn.functional.batch_norm(
            input_252,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_252 = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_254 = torch.nn.functional.silu(input_253, inplace=True)
        input_253 = None
        input_255 = torch.conv2d(
            input_254,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1056,
        )
        input_254 = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_256 = torch.nn.functional.batch_norm(
            input_255,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_255 = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_257 = torch.nn.functional.silu(input_256, inplace=True)
        input_256 = None
        scale_105 = torch.nn.functional.adaptive_avg_pool2d(input_257, 1)
        scale_106 = torch.conv2d(
            scale_105,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_105 = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_107 = torch.nn.functional.silu(scale_106, inplace=True)
        scale_106 = None
        scale_108 = torch.conv2d(
            scale_107,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_107 = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_109 = torch.sigmoid(scale_108)
        scale_108 = None
        input_258 = scale_109 * input_257
        scale_109 = input_257 = None
        input_259 = torch.conv2d(
            input_258,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_258 = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_260 = torch.nn.functional.batch_norm(
            input_259,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_259 = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_261 = torch.conv2d(
            input_260,
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
        input_262 = torch.nn.functional.batch_norm(
            input_261,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_261 = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_263 = torch.nn.functional.silu(input_262, inplace=True)
        input_262 = None
        input_264 = torch.conv2d(
            input_263,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        input_263 = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_265 = torch.nn.functional.batch_norm(
            input_264,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_264 = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_266 = torch.nn.functional.silu(input_265, inplace=True)
        input_265 = None
        scale_110 = torch.nn.functional.adaptive_avg_pool2d(input_266, 1)
        scale_111 = torch.conv2d(
            scale_110,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_110 = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_112 = torch.nn.functional.silu(scale_111, inplace=True)
        scale_111 = None
        scale_113 = torch.conv2d(
            scale_112,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_112 = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_114 = torch.sigmoid(scale_113)
        scale_113 = None
        input_267 = scale_114 * input_266
        scale_114 = input_266 = None
        input_268 = torch.conv2d(
            input_267,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_267 = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_269 = torch.nn.functional.batch_norm(
            input_268,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_268 = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_30 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_30 = None
        input_269 += input_260
        result_30 = input_269
        input_269 = input_260 = None
        input_270 = torch.conv2d(
            result_30,
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
        input_271 = torch.nn.functional.batch_norm(
            input_270,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_270 = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_272 = torch.nn.functional.silu(input_271, inplace=True)
        input_271 = None
        input_273 = torch.conv2d(
            input_272,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        input_272 = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_274 = torch.nn.functional.batch_norm(
            input_273,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_273 = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_275 = torch.nn.functional.silu(input_274, inplace=True)
        input_274 = None
        scale_115 = torch.nn.functional.adaptive_avg_pool2d(input_275, 1)
        scale_116 = torch.conv2d(
            scale_115,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_115 = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_117 = torch.nn.functional.silu(scale_116, inplace=True)
        scale_116 = None
        scale_118 = torch.conv2d(
            scale_117,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_117 = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_119 = torch.sigmoid(scale_118)
        scale_118 = None
        input_276 = scale_119 * input_275
        scale_119 = input_275 = None
        input_277 = torch.conv2d(
            input_276,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_276 = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_278 = torch.nn.functional.batch_norm(
            input_277,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_277 = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_31 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_31 = None
        input_278 += result_30
        result_31 = input_278
        input_278 = result_30 = None
        input_279 = torch.conv2d(
            result_31,
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
        input_280 = torch.nn.functional.batch_norm(
            input_279,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_279 = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_281 = torch.nn.functional.silu(input_280, inplace=True)
        input_280 = None
        input_282 = torch.conv2d(
            input_281,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        input_281 = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_283 = torch.nn.functional.batch_norm(
            input_282,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_282 = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_284 = torch.nn.functional.silu(input_283, inplace=True)
        input_283 = None
        scale_120 = torch.nn.functional.adaptive_avg_pool2d(input_284, 1)
        scale_121 = torch.conv2d(
            scale_120,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_120 = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_122 = torch.nn.functional.silu(scale_121, inplace=True)
        scale_121 = None
        scale_123 = torch.conv2d(
            scale_122,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_122 = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_124 = torch.sigmoid(scale_123)
        scale_123 = None
        input_285 = scale_124 * input_284
        scale_124 = input_284 = None
        input_286 = torch.conv2d(
            input_285,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_285 = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_287 = torch.nn.functional.batch_norm(
            input_286,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_286 = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_32 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_32 = None
        input_287 += result_31
        result_32 = input_287
        input_287 = result_31 = None
        input_288 = torch.conv2d(
            result_32,
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
        input_289 = torch.nn.functional.batch_norm(
            input_288,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_288 = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_290 = torch.nn.functional.silu(input_289, inplace=True)
        input_289 = None
        input_291 = torch.conv2d(
            input_290,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        input_290 = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_292 = torch.nn.functional.batch_norm(
            input_291,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_291 = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_293 = torch.nn.functional.silu(input_292, inplace=True)
        input_292 = None
        scale_125 = torch.nn.functional.adaptive_avg_pool2d(input_293, 1)
        scale_126 = torch.conv2d(
            scale_125,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_125 = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_127 = torch.nn.functional.silu(scale_126, inplace=True)
        scale_126 = None
        scale_128 = torch.conv2d(
            scale_127,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_127 = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_129 = torch.sigmoid(scale_128)
        scale_128 = None
        input_294 = scale_129 * input_293
        scale_129 = input_293 = None
        input_295 = torch.conv2d(
            input_294,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_294 = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_296 = torch.nn.functional.batch_norm(
            input_295,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_295 = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_33 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_33 = None
        input_296 += result_32
        result_33 = input_296
        input_296 = result_32 = None
        input_297 = torch.conv2d(
            result_33,
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
        input_298 = torch.nn.functional.batch_norm(
            input_297,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_297 = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_299 = torch.nn.functional.silu(input_298, inplace=True)
        input_298 = None
        input_300 = torch.conv2d(
            input_299,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        input_299 = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_301 = torch.nn.functional.batch_norm(
            input_300,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_300 = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_302 = torch.nn.functional.silu(input_301, inplace=True)
        input_301 = None
        scale_130 = torch.nn.functional.adaptive_avg_pool2d(input_302, 1)
        scale_131 = torch.conv2d(
            scale_130,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_130 = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_132 = torch.nn.functional.silu(scale_131, inplace=True)
        scale_131 = None
        scale_133 = torch.conv2d(
            scale_132,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_132 = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_134 = torch.sigmoid(scale_133)
        scale_133 = None
        input_303 = scale_134 * input_302
        scale_134 = input_302 = None
        input_304 = torch.conv2d(
            input_303,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_303 = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_305 = torch.nn.functional.batch_norm(
            input_304,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_304 = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_34 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_34 = None
        input_305 += result_33
        result_34 = input_305
        input_305 = result_33 = None
        input_306 = torch.conv2d(
            result_34,
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
        input_307 = torch.nn.functional.batch_norm(
            input_306,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_306 = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_308 = torch.nn.functional.silu(input_307, inplace=True)
        input_307 = None
        input_309 = torch.conv2d(
            input_308,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        input_308 = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_310 = torch.nn.functional.batch_norm(
            input_309,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_309 = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_311 = torch.nn.functional.silu(input_310, inplace=True)
        input_310 = None
        scale_135 = torch.nn.functional.adaptive_avg_pool2d(input_311, 1)
        scale_136 = torch.conv2d(
            scale_135,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_135 = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_137 = torch.nn.functional.silu(scale_136, inplace=True)
        scale_136 = None
        scale_138 = torch.conv2d(
            scale_137,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_137 = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_139 = torch.sigmoid(scale_138)
        scale_138 = None
        input_312 = scale_139 * input_311
        scale_139 = input_311 = None
        input_313 = torch.conv2d(
            input_312,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_312 = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_314 = torch.nn.functional.batch_norm(
            input_313,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_313 = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_35 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_35 = None
        input_314 += result_34
        result_35 = input_314
        input_314 = result_34 = None
        input_315 = torch.conv2d(
            result_35,
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
        input_316 = torch.nn.functional.batch_norm(
            input_315,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_315 = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_317 = torch.nn.functional.silu(input_316, inplace=True)
        input_316 = None
        input_318 = torch.conv2d(
            input_317,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        input_317 = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_319 = torch.nn.functional.batch_norm(
            input_318,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_318 = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_320 = torch.nn.functional.silu(input_319, inplace=True)
        input_319 = None
        scale_140 = torch.nn.functional.adaptive_avg_pool2d(input_320, 1)
        scale_141 = torch.conv2d(
            scale_140,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_140 = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_142 = torch.nn.functional.silu(scale_141, inplace=True)
        scale_141 = None
        scale_143 = torch.conv2d(
            scale_142,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_142 = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_144 = torch.sigmoid(scale_143)
        scale_143 = None
        input_321 = scale_144 * input_320
        scale_144 = input_320 = None
        input_322 = torch.conv2d(
            input_321,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_321 = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_323 = torch.nn.functional.batch_norm(
            input_322,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_322 = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_36 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_36 = None
        input_323 += result_35
        result_36 = input_323
        input_323 = result_35 = None
        input_324 = torch.conv2d(
            result_36,
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
        input_325 = torch.nn.functional.batch_norm(
            input_324,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_324 = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_326 = torch.nn.functional.silu(input_325, inplace=True)
        input_325 = None
        input_327 = torch.conv2d(
            input_326,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        input_326 = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_328 = torch.nn.functional.batch_norm(
            input_327,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_327 = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_329 = torch.nn.functional.silu(input_328, inplace=True)
        input_328 = None
        scale_145 = torch.nn.functional.adaptive_avg_pool2d(input_329, 1)
        scale_146 = torch.conv2d(
            scale_145,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_145 = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_147 = torch.nn.functional.silu(scale_146, inplace=True)
        scale_146 = None
        scale_148 = torch.conv2d(
            scale_147,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_147 = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_149 = torch.sigmoid(scale_148)
        scale_148 = None
        input_330 = scale_149 * input_329
        scale_149 = input_329 = None
        input_331 = torch.conv2d(
            input_330,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_330 = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_332 = torch.nn.functional.batch_norm(
            input_331,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_331 = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_37 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_37 = None
        input_332 += result_36
        result_37 = input_332
        input_332 = result_36 = None
        input_333 = torch.conv2d(
            result_37,
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
        input_334 = torch.nn.functional.batch_norm(
            input_333,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_333 = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_335 = torch.nn.functional.silu(input_334, inplace=True)
        input_334 = None
        input_336 = torch.conv2d(
            input_335,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        input_335 = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_337 = torch.nn.functional.batch_norm(
            input_336,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_336 = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_338 = torch.nn.functional.silu(input_337, inplace=True)
        input_337 = None
        scale_150 = torch.nn.functional.adaptive_avg_pool2d(input_338, 1)
        scale_151 = torch.conv2d(
            scale_150,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_150 = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_152 = torch.nn.functional.silu(scale_151, inplace=True)
        scale_151 = None
        scale_153 = torch.conv2d(
            scale_152,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_152 = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_154 = torch.sigmoid(scale_153)
        scale_153 = None
        input_339 = scale_154 * input_338
        scale_154 = input_338 = None
        input_340 = torch.conv2d(
            input_339,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_339 = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_341 = torch.nn.functional.batch_norm(
            input_340,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_340 = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_38 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_38 = None
        input_341 += result_37
        result_38 = input_341
        input_341 = result_37 = None
        input_342 = torch.conv2d(
            result_38,
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
        input_343 = torch.nn.functional.batch_norm(
            input_342,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_342 = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_344 = torch.nn.functional.silu(input_343, inplace=True)
        input_343 = None
        input_345 = torch.conv2d(
            input_344,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        input_344 = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_346 = torch.nn.functional.batch_norm(
            input_345,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_345 = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_347 = torch.nn.functional.silu(input_346, inplace=True)
        input_346 = None
        scale_155 = torch.nn.functional.adaptive_avg_pool2d(input_347, 1)
        scale_156 = torch.conv2d(
            scale_155,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_155 = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_157 = torch.nn.functional.silu(scale_156, inplace=True)
        scale_156 = None
        scale_158 = torch.conv2d(
            scale_157,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_157 = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_159 = torch.sigmoid(scale_158)
        scale_158 = None
        input_348 = scale_159 * input_347
        scale_159 = input_347 = None
        input_349 = torch.conv2d(
            input_348,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_348 = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_350 = torch.nn.functional.batch_norm(
            input_349,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_349 = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_39 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_39 = None
        input_350 += result_38
        result_39 = input_350
        input_350 = result_38 = None
        input_351 = torch.conv2d(
            result_39,
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
        input_352 = torch.nn.functional.batch_norm(
            input_351,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_351 = l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_353 = torch.nn.functional.silu(input_352, inplace=True)
        input_352 = None
        input_354 = torch.conv2d(
            input_353,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        input_353 = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_355 = torch.nn.functional.batch_norm(
            input_354,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_354 = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_356 = torch.nn.functional.silu(input_355, inplace=True)
        input_355 = None
        scale_160 = torch.nn.functional.adaptive_avg_pool2d(input_356, 1)
        scale_161 = torch.conv2d(
            scale_160,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_160 = l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_162 = torch.nn.functional.silu(scale_161, inplace=True)
        scale_161 = None
        scale_163 = torch.conv2d(
            scale_162,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_162 = l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_164 = torch.sigmoid(scale_163)
        scale_163 = None
        input_357 = scale_164 * input_356
        scale_164 = input_356 = None
        input_358 = torch.conv2d(
            input_357,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_357 = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_359 = torch.nn.functional.batch_norm(
            input_358,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_358 = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_40 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_40 = None
        input_359 += result_39
        result_40 = input_359
        input_359 = result_39 = None
        input_360 = torch.conv2d(
            result_40,
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
        input_361 = torch.nn.functional.batch_norm(
            input_360,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_360 = l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_362 = torch.nn.functional.silu(input_361, inplace=True)
        input_361 = None
        input_363 = torch.conv2d(
            input_362,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        input_362 = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_364 = torch.nn.functional.batch_norm(
            input_363,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_363 = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_365 = torch.nn.functional.silu(input_364, inplace=True)
        input_364 = None
        scale_165 = torch.nn.functional.adaptive_avg_pool2d(input_365, 1)
        scale_166 = torch.conv2d(
            scale_165,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_165 = l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_167 = torch.nn.functional.silu(scale_166, inplace=True)
        scale_166 = None
        scale_168 = torch.conv2d(
            scale_167,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_167 = l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_169 = torch.sigmoid(scale_168)
        scale_168 = None
        input_366 = scale_169 * input_365
        scale_169 = input_365 = None
        input_367 = torch.conv2d(
            input_366,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_366 = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_368 = torch.nn.functional.batch_norm(
            input_367,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_367 = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_41 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_41 = None
        input_368 += result_40
        result_41 = input_368
        input_368 = result_40 = None
        input_369 = torch.conv2d(
            result_41,
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
        input_370 = torch.nn.functional.batch_norm(
            input_369,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_369 = l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_371 = torch.nn.functional.silu(input_370, inplace=True)
        input_370 = None
        input_372 = torch.conv2d(
            input_371,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        input_371 = l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_373 = torch.nn.functional.batch_norm(
            input_372,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_372 = l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_374 = torch.nn.functional.silu(input_373, inplace=True)
        input_373 = None
        scale_170 = torch.nn.functional.adaptive_avg_pool2d(input_374, 1)
        scale_171 = torch.conv2d(
            scale_170,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_170 = l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_172 = torch.nn.functional.silu(scale_171, inplace=True)
        scale_171 = None
        scale_173 = torch.conv2d(
            scale_172,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_172 = l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_174 = torch.sigmoid(scale_173)
        scale_173 = None
        input_375 = scale_174 * input_374
        scale_174 = input_374 = None
        input_376 = torch.conv2d(
            input_375,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_375 = l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_377 = torch.nn.functional.batch_norm(
            input_376,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_376 = l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_42 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_42 = None
        input_377 += result_41
        result_42 = input_377
        input_377 = result_41 = None
        input_378 = torch.conv2d(
            result_42,
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
        input_379 = torch.nn.functional.batch_norm(
            input_378,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_378 = l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_380 = torch.nn.functional.silu(input_379, inplace=True)
        input_379 = None
        input_381 = torch.conv2d(
            input_380,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        input_380 = l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_382 = torch.nn.functional.batch_norm(
            input_381,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_381 = l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_383 = torch.nn.functional.silu(input_382, inplace=True)
        input_382 = None
        scale_175 = torch.nn.functional.adaptive_avg_pool2d(input_383, 1)
        scale_176 = torch.conv2d(
            scale_175,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_175 = l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_177 = torch.nn.functional.silu(scale_176, inplace=True)
        scale_176 = None
        scale_178 = torch.conv2d(
            scale_177,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_177 = l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_179 = torch.sigmoid(scale_178)
        scale_178 = None
        input_384 = scale_179 * input_383
        scale_179 = input_383 = None
        input_385 = torch.conv2d(
            input_384,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_384 = l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_386 = torch.nn.functional.batch_norm(
            input_385,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_385 = l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_43 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_43 = None
        input_386 += result_42
        result_43 = input_386
        input_386 = result_42 = None
        input_387 = torch.conv2d(
            result_43,
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
        input_388 = torch.nn.functional.batch_norm(
            input_387,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_387 = l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_389 = torch.nn.functional.silu(input_388, inplace=True)
        input_388 = None
        input_390 = torch.conv2d(
            input_389,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        input_389 = l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_391 = torch.nn.functional.batch_norm(
            input_390,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_390 = l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_392 = torch.nn.functional.silu(input_391, inplace=True)
        input_391 = None
        scale_180 = torch.nn.functional.adaptive_avg_pool2d(input_392, 1)
        scale_181 = torch.conv2d(
            scale_180,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_180 = l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_182 = torch.nn.functional.silu(scale_181, inplace=True)
        scale_181 = None
        scale_183 = torch.conv2d(
            scale_182,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_182 = l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_184 = torch.sigmoid(scale_183)
        scale_183 = None
        input_393 = scale_184 * input_392
        scale_184 = input_392 = None
        input_394 = torch.conv2d(
            input_393,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_393 = l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_395 = torch.nn.functional.batch_norm(
            input_394,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_394 = l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_15_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_44 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_44 = None
        input_395 += result_43
        result_44 = input_395
        input_395 = result_43 = None
        input_396 = torch.conv2d(
            result_44,
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
        input_397 = torch.nn.functional.batch_norm(
            input_396,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_396 = l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_398 = torch.nn.functional.silu(input_397, inplace=True)
        input_397 = None
        input_399 = torch.conv2d(
            input_398,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        input_398 = l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_400 = torch.nn.functional.batch_norm(
            input_399,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_399 = l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_401 = torch.nn.functional.silu(input_400, inplace=True)
        input_400 = None
        scale_185 = torch.nn.functional.adaptive_avg_pool2d(input_401, 1)
        scale_186 = torch.conv2d(
            scale_185,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_185 = l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_187 = torch.nn.functional.silu(scale_186, inplace=True)
        scale_186 = None
        scale_188 = torch.conv2d(
            scale_187,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_187 = l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_189 = torch.sigmoid(scale_188)
        scale_188 = None
        input_402 = scale_189 * input_401
        scale_189 = input_401 = None
        input_403 = torch.conv2d(
            input_402,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_402 = l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_404 = torch.nn.functional.batch_norm(
            input_403,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_403 = l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_16_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_45 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_45 = None
        input_404 += result_44
        result_45 = input_404
        input_404 = result_44 = None
        input_405 = torch.conv2d(
            result_45,
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
        input_406 = torch.nn.functional.batch_norm(
            input_405,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_405 = l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_407 = torch.nn.functional.silu(input_406, inplace=True)
        input_406 = None
        input_408 = torch.conv2d(
            input_407,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        input_407 = l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_409 = torch.nn.functional.batch_norm(
            input_408,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_408 = l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_410 = torch.nn.functional.silu(input_409, inplace=True)
        input_409 = None
        scale_190 = torch.nn.functional.adaptive_avg_pool2d(input_410, 1)
        scale_191 = torch.conv2d(
            scale_190,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_190 = l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_192 = torch.nn.functional.silu(scale_191, inplace=True)
        scale_191 = None
        scale_193 = torch.conv2d(
            scale_192,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_192 = l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_194 = torch.sigmoid(scale_193)
        scale_193 = None
        input_411 = scale_194 * input_410
        scale_194 = input_410 = None
        input_412 = torch.conv2d(
            input_411,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_411 = l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_413 = torch.nn.functional.batch_norm(
            input_412,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_412 = l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_17_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_46 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_46 = None
        input_413 += result_45
        result_46 = input_413
        input_413 = result_45 = None
        input_414 = torch.conv2d(
            result_46,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_46 = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_415 = torch.nn.functional.batch_norm(
            input_414,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_414 = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_416 = torch.nn.functional.silu(input_415, inplace=True)
        input_415 = None
        input_417 = torch.conv2d(
            input_416,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1824,
        )
        input_416 = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_418 = torch.nn.functional.batch_norm(
            input_417,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_417 = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_419 = torch.nn.functional.silu(input_418, inplace=True)
        input_418 = None
        scale_195 = torch.nn.functional.adaptive_avg_pool2d(input_419, 1)
        scale_196 = torch.conv2d(
            scale_195,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_195 = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_197 = torch.nn.functional.silu(scale_196, inplace=True)
        scale_196 = None
        scale_198 = torch.conv2d(
            scale_197,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_197 = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_199 = torch.sigmoid(scale_198)
        scale_198 = None
        input_420 = scale_199 * input_419
        scale_199 = input_419 = None
        input_421 = torch.conv2d(
            input_420,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_420 = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_422 = torch.nn.functional.batch_norm(
            input_421,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_421 = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_423 = torch.conv2d(
            input_422,
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
        input_424 = torch.nn.functional.batch_norm(
            input_423,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_423 = l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_425 = torch.nn.functional.silu(input_424, inplace=True)
        input_424 = None
        input_426 = torch.conv2d(
            input_425,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3072,
        )
        input_425 = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_427 = torch.nn.functional.batch_norm(
            input_426,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_426 = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_428 = torch.nn.functional.silu(input_427, inplace=True)
        input_427 = None
        scale_200 = torch.nn.functional.adaptive_avg_pool2d(input_428, 1)
        scale_201 = torch.conv2d(
            scale_200,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_200 = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_202 = torch.nn.functional.silu(scale_201, inplace=True)
        scale_201 = None
        scale_203 = torch.conv2d(
            scale_202,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_202 = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_204 = torch.sigmoid(scale_203)
        scale_203 = None
        input_429 = scale_204 * input_428
        scale_204 = input_428 = None
        input_430 = torch.conv2d(
            input_429,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_429 = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_431 = torch.nn.functional.batch_norm(
            input_430,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_430 = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_47 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_47 = None
        input_431 += input_422
        result_47 = input_431
        input_431 = input_422 = None
        input_432 = torch.conv2d(
            result_47,
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
        input_433 = torch.nn.functional.batch_norm(
            input_432,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_432 = l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_434 = torch.nn.functional.silu(input_433, inplace=True)
        input_433 = None
        input_435 = torch.conv2d(
            input_434,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3072,
        )
        input_434 = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_436 = torch.nn.functional.batch_norm(
            input_435,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_435 = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_437 = torch.nn.functional.silu(input_436, inplace=True)
        input_436 = None
        scale_205 = torch.nn.functional.adaptive_avg_pool2d(input_437, 1)
        scale_206 = torch.conv2d(
            scale_205,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_205 = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_207 = torch.nn.functional.silu(scale_206, inplace=True)
        scale_206 = None
        scale_208 = torch.conv2d(
            scale_207,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_207 = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_209 = torch.sigmoid(scale_208)
        scale_208 = None
        input_438 = scale_209 * input_437
        scale_209 = input_437 = None
        input_439 = torch.conv2d(
            input_438,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_438 = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_440 = torch.nn.functional.batch_norm(
            input_439,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_439 = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_48 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_48 = None
        input_440 += result_47
        result_48 = input_440
        input_440 = result_47 = None
        input_441 = torch.conv2d(
            result_48,
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
        input_442 = torch.nn.functional.batch_norm(
            input_441,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_441 = l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_443 = torch.nn.functional.silu(input_442, inplace=True)
        input_442 = None
        input_444 = torch.conv2d(
            input_443,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3072,
        )
        input_443 = l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_445 = torch.nn.functional.batch_norm(
            input_444,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_444 = l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_446 = torch.nn.functional.silu(input_445, inplace=True)
        input_445 = None
        scale_210 = torch.nn.functional.adaptive_avg_pool2d(input_446, 1)
        scale_211 = torch.conv2d(
            scale_210,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_210 = l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_212 = torch.nn.functional.silu(scale_211, inplace=True)
        scale_211 = None
        scale_213 = torch.conv2d(
            scale_212,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_212 = l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_214 = torch.sigmoid(scale_213)
        scale_213 = None
        input_447 = scale_214 * input_446
        scale_214 = input_446 = None
        input_448 = torch.conv2d(
            input_447,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_447 = l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_449 = torch.nn.functional.batch_norm(
            input_448,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_448 = l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_49 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_49 = None
        input_449 += result_48
        result_49 = input_449
        input_449 = result_48 = None
        input_450 = torch.conv2d(
            result_49,
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
        input_451 = torch.nn.functional.batch_norm(
            input_450,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_450 = l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_452 = torch.nn.functional.silu(input_451, inplace=True)
        input_451 = None
        input_453 = torch.conv2d(
            input_452,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3072,
        )
        input_452 = l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_454 = torch.nn.functional.batch_norm(
            input_453,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_453 = l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_455 = torch.nn.functional.silu(input_454, inplace=True)
        input_454 = None
        scale_215 = torch.nn.functional.adaptive_avg_pool2d(input_455, 1)
        scale_216 = torch.conv2d(
            scale_215,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_215 = l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_217 = torch.nn.functional.silu(scale_216, inplace=True)
        scale_216 = None
        scale_218 = torch.conv2d(
            scale_217,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_217 = l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_219 = torch.sigmoid(scale_218)
        scale_218 = None
        input_456 = scale_219 * input_455
        scale_219 = input_455 = None
        input_457 = torch.conv2d(
            input_456,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_456 = l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_458 = torch.nn.functional.batch_norm(
            input_457,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_457 = l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_50 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_50 = None
        input_458 += result_49
        result_50 = input_458
        input_458 = result_49 = None
        input_459 = torch.conv2d(
            result_50,
            l_self_modules_features_modules_8_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_50 = (
            l_self_modules_features_modules_8_modules_0_parameters_weight_
        ) = None
        input_460 = torch.nn.functional.batch_norm(
            input_459,
            l_self_modules_features_modules_8_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_1_buffers_running_var_,
            l_self_modules_features_modules_8_modules_1_parameters_weight_,
            l_self_modules_features_modules_8_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_459 = (
            l_self_modules_features_modules_8_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_8_modules_1_buffers_running_var_
        ) = (
            l_self_modules_features_modules_8_modules_1_parameters_weight_
        ) = l_self_modules_features_modules_8_modules_1_parameters_bias_ = None
        input_461 = torch.nn.functional.silu(input_460, inplace=True)
        input_460 = None
        x = torch.nn.functional.adaptive_avg_pool2d(input_461, 1)
        input_461 = None
        x_1 = torch.flatten(x, 1)
        x = None
        input_462 = torch.nn.functional.dropout(x_1, 0.3, False, True)
        x_1 = None
        input_463 = torch._C._nn.linear(
            input_462,
            l_self_modules_classifier_modules_1_parameters_weight_,
            l_self_modules_classifier_modules_1_parameters_bias_,
        )
        input_462 = (
            l_self_modules_classifier_modules_1_parameters_weight_
        ) = l_self_modules_classifier_modules_1_parameters_bias_ = None
        return (input_463,)
