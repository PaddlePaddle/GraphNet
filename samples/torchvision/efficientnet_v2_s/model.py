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
        L_self_modules_features_modules_7_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_features_modules_7_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_7_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_1_buffers_running_mean_ = (
            L_self_modules_features_modules_7_modules_1_buffers_running_mean_
        )
        l_self_modules_features_modules_7_modules_1_buffers_running_var_ = (
            L_self_modules_features_modules_7_modules_1_buffers_running_var_
        )
        l_self_modules_features_modules_7_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_7_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_7_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_7_modules_1_parameters_bias_
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
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        result_1 = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_11 = torch.nn.functional.batch_norm(
            input_10,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_10 = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_12 = torch.nn.functional.silu(input_11, inplace=True)
        input_11 = None
        input_13 = torch.conv2d(
            input_12,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_12 = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_14 = torch.nn.functional.batch_norm(
            input_13,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_13 = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_15 = torch.conv2d(
            input_14,
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
        input_16 = torch.nn.functional.batch_norm(
            input_15,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_15 = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_17 = torch.nn.functional.silu(input_16, inplace=True)
        input_16 = None
        input_18 = torch.conv2d(
            input_17,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_17 = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_19 = torch.nn.functional.batch_norm(
            input_18,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_18 = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_2 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_2 = None
        input_19 += input_14
        result_2 = input_19
        input_19 = input_14 = None
        input_20 = torch.conv2d(
            result_2,
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
        input_21 = torch.nn.functional.batch_norm(
            input_20,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_20 = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_22 = torch.nn.functional.silu(input_21, inplace=True)
        input_21 = None
        input_23 = torch.conv2d(
            input_22,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_22 = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_24 = torch.nn.functional.batch_norm(
            input_23,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_23 = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_3 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_3 = None
        input_24 += result_2
        result_3 = input_24
        input_24 = result_2 = None
        input_25 = torch.conv2d(
            result_3,
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
        input_26 = torch.nn.functional.batch_norm(
            input_25,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_25 = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_27 = torch.nn.functional.silu(input_26, inplace=True)
        input_26 = None
        input_28 = torch.conv2d(
            input_27,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_27 = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_29 = torch.nn.functional.batch_norm(
            input_28,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_28 = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_4 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_4 = None
        input_29 += result_3
        result_4 = input_29
        input_29 = result_3 = None
        input_30 = torch.conv2d(
            result_4,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        result_4 = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_31 = torch.nn.functional.batch_norm(
            input_30,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_30 = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_32 = torch.nn.functional.silu(input_31, inplace=True)
        input_31 = None
        input_33 = torch.conv2d(
            input_32,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_32 = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_34 = torch.nn.functional.batch_norm(
            input_33,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_33 = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_35 = torch.conv2d(
            input_34,
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
        input_36 = torch.nn.functional.batch_norm(
            input_35,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_35 = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_37 = torch.nn.functional.silu(input_36, inplace=True)
        input_36 = None
        input_38 = torch.conv2d(
            input_37,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_37 = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_39 = torch.nn.functional.batch_norm(
            input_38,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_38 = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_5 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_5 = None
        input_39 += input_34
        result_5 = input_39
        input_39 = input_34 = None
        input_40 = torch.conv2d(
            result_5,
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
        input_41 = torch.nn.functional.batch_norm(
            input_40,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_40 = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_42 = torch.nn.functional.silu(input_41, inplace=True)
        input_41 = None
        input_43 = torch.conv2d(
            input_42,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_42 = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_44 = torch.nn.functional.batch_norm(
            input_43,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_43 = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_6 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_6 = None
        input_44 += result_5
        result_6 = input_44
        input_44 = result_5 = None
        input_45 = torch.conv2d(
            result_6,
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
        input_46 = torch.nn.functional.batch_norm(
            input_45,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_45 = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_47 = torch.nn.functional.silu(input_46, inplace=True)
        input_46 = None
        input_48 = torch.conv2d(
            input_47,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_47 = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_49 = torch.nn.functional.batch_norm(
            input_48,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_48 = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_7 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_7 = None
        input_49 += result_6
        result_7 = input_49
        input_49 = result_6 = None
        input_50 = torch.conv2d(
            result_7,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_7 = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_51 = torch.nn.functional.batch_norm(
            input_50,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_50 = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_52 = torch.nn.functional.silu(input_51, inplace=True)
        input_51 = None
        input_53 = torch.conv2d(
            input_52,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            256,
        )
        input_52 = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_54 = torch.nn.functional.batch_norm(
            input_53,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_53 = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_55 = torch.nn.functional.silu(input_54, inplace=True)
        input_54 = None
        scale = torch.nn.functional.adaptive_avg_pool2d(input_55, 1)
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
        input_56 = scale_4 * input_55
        scale_4 = input_55 = None
        input_57 = torch.conv2d(
            input_56,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_56 = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_58 = torch.nn.functional.batch_norm(
            input_57,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_57 = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_59 = torch.conv2d(
            input_58,
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
        input_60 = torch.nn.functional.batch_norm(
            input_59,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_59 = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_61 = torch.nn.functional.silu(input_60, inplace=True)
        input_60 = None
        input_62 = torch.conv2d(
            input_61,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        input_61 = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_63 = torch.nn.functional.batch_norm(
            input_62,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_62 = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_64 = torch.nn.functional.silu(input_63, inplace=True)
        input_63 = None
        scale_5 = torch.nn.functional.adaptive_avg_pool2d(input_64, 1)
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
        input_65 = scale_9 * input_64
        scale_9 = input_64 = None
        input_66 = torch.conv2d(
            input_65,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_65 = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_67 = torch.nn.functional.batch_norm(
            input_66,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_66 = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_8 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_8 = None
        input_67 += input_58
        result_8 = input_67
        input_67 = input_58 = None
        input_68 = torch.conv2d(
            result_8,
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
        input_69 = torch.nn.functional.batch_norm(
            input_68,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_68 = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_70 = torch.nn.functional.silu(input_69, inplace=True)
        input_69 = None
        input_71 = torch.conv2d(
            input_70,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        input_70 = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_72 = torch.nn.functional.batch_norm(
            input_71,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_71 = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_73 = torch.nn.functional.silu(input_72, inplace=True)
        input_72 = None
        scale_10 = torch.nn.functional.adaptive_avg_pool2d(input_73, 1)
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
        input_74 = scale_14 * input_73
        scale_14 = input_73 = None
        input_75 = torch.conv2d(
            input_74,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_74 = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_76 = torch.nn.functional.batch_norm(
            input_75,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_75 = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_9 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_9 = None
        input_76 += result_8
        result_9 = input_76
        input_76 = result_8 = None
        input_77 = torch.conv2d(
            result_9,
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
        input_78 = torch.nn.functional.batch_norm(
            input_77,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_77 = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_79 = torch.nn.functional.silu(input_78, inplace=True)
        input_78 = None
        input_80 = torch.conv2d(
            input_79,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        input_79 = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_81 = torch.nn.functional.batch_norm(
            input_80,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_80 = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_82 = torch.nn.functional.silu(input_81, inplace=True)
        input_81 = None
        scale_15 = torch.nn.functional.adaptive_avg_pool2d(input_82, 1)
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
        input_83 = scale_19 * input_82
        scale_19 = input_82 = None
        input_84 = torch.conv2d(
            input_83,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_83 = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_85 = torch.nn.functional.batch_norm(
            input_84,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_84 = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_10 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_10 = None
        input_85 += result_9
        result_10 = input_85
        input_85 = result_9 = None
        input_86 = torch.conv2d(
            result_10,
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
        input_87 = torch.nn.functional.batch_norm(
            input_86,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_86 = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_88 = torch.nn.functional.silu(input_87, inplace=True)
        input_87 = None
        input_89 = torch.conv2d(
            input_88,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        input_88 = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_90 = torch.nn.functional.batch_norm(
            input_89,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_89 = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_91 = torch.nn.functional.silu(input_90, inplace=True)
        input_90 = None
        scale_20 = torch.nn.functional.adaptive_avg_pool2d(input_91, 1)
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
        input_92 = scale_24 * input_91
        scale_24 = input_91 = None
        input_93 = torch.conv2d(
            input_92,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_92 = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_94 = torch.nn.functional.batch_norm(
            input_93,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_93 = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_11 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_11 = None
        input_94 += result_10
        result_11 = input_94
        input_94 = result_10 = None
        input_95 = torch.conv2d(
            result_11,
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
        input_96 = torch.nn.functional.batch_norm(
            input_95,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_95 = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_97 = torch.nn.functional.silu(input_96, inplace=True)
        input_96 = None
        input_98 = torch.conv2d(
            input_97,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        input_97 = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_99 = torch.nn.functional.batch_norm(
            input_98,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_98 = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_100 = torch.nn.functional.silu(input_99, inplace=True)
        input_99 = None
        scale_25 = torch.nn.functional.adaptive_avg_pool2d(input_100, 1)
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
        input_101 = scale_29 * input_100
        scale_29 = input_100 = None
        input_102 = torch.conv2d(
            input_101,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_101 = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_103 = torch.nn.functional.batch_norm(
            input_102,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_102 = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_12 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_12 = None
        input_103 += result_11
        result_12 = input_103
        input_103 = result_11 = None
        input_104 = torch.conv2d(
            result_12,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_12 = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_105 = torch.nn.functional.batch_norm(
            input_104,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_104 = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_106 = torch.nn.functional.silu(input_105, inplace=True)
        input_105 = None
        input_107 = torch.conv2d(
            input_106,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            768,
        )
        input_106 = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_108 = torch.nn.functional.batch_norm(
            input_107,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_107 = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_109 = torch.nn.functional.silu(input_108, inplace=True)
        input_108 = None
        scale_30 = torch.nn.functional.adaptive_avg_pool2d(input_109, 1)
        scale_31 = torch.conv2d(
            scale_30,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_30 = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_32 = torch.nn.functional.silu(scale_31, inplace=True)
        scale_31 = None
        scale_33 = torch.conv2d(
            scale_32,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_32 = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_34 = torch.sigmoid(scale_33)
        scale_33 = None
        input_110 = scale_34 * input_109
        scale_34 = input_109 = None
        input_111 = torch.conv2d(
            input_110,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_110 = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_112 = torch.nn.functional.batch_norm(
            input_111,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_111 = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_113 = torch.conv2d(
            input_112,
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
        input_114 = torch.nn.functional.batch_norm(
            input_113,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_113 = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_115 = torch.nn.functional.silu(input_114, inplace=True)
        input_114 = None
        input_116 = torch.conv2d(
            input_115,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_115 = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_117 = torch.nn.functional.batch_norm(
            input_116,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_116 = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_118 = torch.nn.functional.silu(input_117, inplace=True)
        input_117 = None
        scale_35 = torch.nn.functional.adaptive_avg_pool2d(input_118, 1)
        scale_36 = torch.conv2d(
            scale_35,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_35 = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_37 = torch.nn.functional.silu(scale_36, inplace=True)
        scale_36 = None
        scale_38 = torch.conv2d(
            scale_37,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_37 = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_39 = torch.sigmoid(scale_38)
        scale_38 = None
        input_119 = scale_39 * input_118
        scale_39 = input_118 = None
        input_120 = torch.conv2d(
            input_119,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_119 = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_121 = torch.nn.functional.batch_norm(
            input_120,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_120 = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_13 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_13 = None
        input_121 += input_112
        result_13 = input_121
        input_121 = input_112 = None
        input_122 = torch.conv2d(
            result_13,
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
        input_123 = torch.nn.functional.batch_norm(
            input_122,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_122 = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_124 = torch.nn.functional.silu(input_123, inplace=True)
        input_123 = None
        input_125 = torch.conv2d(
            input_124,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_124 = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_126 = torch.nn.functional.batch_norm(
            input_125,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_125 = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_127 = torch.nn.functional.silu(input_126, inplace=True)
        input_126 = None
        scale_40 = torch.nn.functional.adaptive_avg_pool2d(input_127, 1)
        scale_41 = torch.conv2d(
            scale_40,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_40 = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_42 = torch.nn.functional.silu(scale_41, inplace=True)
        scale_41 = None
        scale_43 = torch.conv2d(
            scale_42,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_42 = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_44 = torch.sigmoid(scale_43)
        scale_43 = None
        input_128 = scale_44 * input_127
        scale_44 = input_127 = None
        input_129 = torch.conv2d(
            input_128,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_128 = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_130 = torch.nn.functional.batch_norm(
            input_129,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_129 = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_14 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_14 = None
        input_130 += result_13
        result_14 = input_130
        input_130 = result_13 = None
        input_131 = torch.conv2d(
            result_14,
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
        input_132 = torch.nn.functional.batch_norm(
            input_131,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_131 = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_133 = torch.nn.functional.silu(input_132, inplace=True)
        input_132 = None
        input_134 = torch.conv2d(
            input_133,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_133 = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_135 = torch.nn.functional.batch_norm(
            input_134,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_134 = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_136 = torch.nn.functional.silu(input_135, inplace=True)
        input_135 = None
        scale_45 = torch.nn.functional.adaptive_avg_pool2d(input_136, 1)
        scale_46 = torch.conv2d(
            scale_45,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_45 = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_47 = torch.nn.functional.silu(scale_46, inplace=True)
        scale_46 = None
        scale_48 = torch.conv2d(
            scale_47,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_47 = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_49 = torch.sigmoid(scale_48)
        scale_48 = None
        input_137 = scale_49 * input_136
        scale_49 = input_136 = None
        input_138 = torch.conv2d(
            input_137,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_137 = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_139 = torch.nn.functional.batch_norm(
            input_138,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_138 = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_15 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_15 = None
        input_139 += result_14
        result_15 = input_139
        input_139 = result_14 = None
        input_140 = torch.conv2d(
            result_15,
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
        input_141 = torch.nn.functional.batch_norm(
            input_140,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_140 = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_142 = torch.nn.functional.silu(input_141, inplace=True)
        input_141 = None
        input_143 = torch.conv2d(
            input_142,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_142 = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_144 = torch.nn.functional.batch_norm(
            input_143,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_143 = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_145 = torch.nn.functional.silu(input_144, inplace=True)
        input_144 = None
        scale_50 = torch.nn.functional.adaptive_avg_pool2d(input_145, 1)
        scale_51 = torch.conv2d(
            scale_50,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_50 = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_52 = torch.nn.functional.silu(scale_51, inplace=True)
        scale_51 = None
        scale_53 = torch.conv2d(
            scale_52,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_52 = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_54 = torch.sigmoid(scale_53)
        scale_53 = None
        input_146 = scale_54 * input_145
        scale_54 = input_145 = None
        input_147 = torch.conv2d(
            input_146,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_146 = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_148 = torch.nn.functional.batch_norm(
            input_147,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_147 = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_16 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_16 = None
        input_148 += result_15
        result_16 = input_148
        input_148 = result_15 = None
        input_149 = torch.conv2d(
            result_16,
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
        input_150 = torch.nn.functional.batch_norm(
            input_149,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_149 = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_151 = torch.nn.functional.silu(input_150, inplace=True)
        input_150 = None
        input_152 = torch.conv2d(
            input_151,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_151 = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_153 = torch.nn.functional.batch_norm(
            input_152,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_152 = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_154 = torch.nn.functional.silu(input_153, inplace=True)
        input_153 = None
        scale_55 = torch.nn.functional.adaptive_avg_pool2d(input_154, 1)
        scale_56 = torch.conv2d(
            scale_55,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_55 = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_57 = torch.nn.functional.silu(scale_56, inplace=True)
        scale_56 = None
        scale_58 = torch.conv2d(
            scale_57,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_57 = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_59 = torch.sigmoid(scale_58)
        scale_58 = None
        input_155 = scale_59 * input_154
        scale_59 = input_154 = None
        input_156 = torch.conv2d(
            input_155,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_155 = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_157 = torch.nn.functional.batch_norm(
            input_156,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_156 = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_17 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_17 = None
        input_157 += result_16
        result_17 = input_157
        input_157 = result_16 = None
        input_158 = torch.conv2d(
            result_17,
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
        input_159 = torch.nn.functional.batch_norm(
            input_158,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_158 = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_160 = torch.nn.functional.silu(input_159, inplace=True)
        input_159 = None
        input_161 = torch.conv2d(
            input_160,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_160 = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_162 = torch.nn.functional.batch_norm(
            input_161,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_161 = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_163 = torch.nn.functional.silu(input_162, inplace=True)
        input_162 = None
        scale_60 = torch.nn.functional.adaptive_avg_pool2d(input_163, 1)
        scale_61 = torch.conv2d(
            scale_60,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_60 = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_62 = torch.nn.functional.silu(scale_61, inplace=True)
        scale_61 = None
        scale_63 = torch.conv2d(
            scale_62,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_62 = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_64 = torch.sigmoid(scale_63)
        scale_63 = None
        input_164 = scale_64 * input_163
        scale_64 = input_163 = None
        input_165 = torch.conv2d(
            input_164,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_164 = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_166 = torch.nn.functional.batch_norm(
            input_165,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_165 = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_18 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_18 = None
        input_166 += result_17
        result_18 = input_166
        input_166 = result_17 = None
        input_167 = torch.conv2d(
            result_18,
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
        input_168 = torch.nn.functional.batch_norm(
            input_167,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_167 = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_169 = torch.nn.functional.silu(input_168, inplace=True)
        input_168 = None
        input_170 = torch.conv2d(
            input_169,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_169 = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_171 = torch.nn.functional.batch_norm(
            input_170,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_170 = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_172 = torch.nn.functional.silu(input_171, inplace=True)
        input_171 = None
        scale_65 = torch.nn.functional.adaptive_avg_pool2d(input_172, 1)
        scale_66 = torch.conv2d(
            scale_65,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_65 = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_67 = torch.nn.functional.silu(scale_66, inplace=True)
        scale_66 = None
        scale_68 = torch.conv2d(
            scale_67,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_67 = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_69 = torch.sigmoid(scale_68)
        scale_68 = None
        input_173 = scale_69 * input_172
        scale_69 = input_172 = None
        input_174 = torch.conv2d(
            input_173,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_173 = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_175 = torch.nn.functional.batch_norm(
            input_174,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_174 = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_19 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_19 = None
        input_175 += result_18
        result_19 = input_175
        input_175 = result_18 = None
        input_176 = torch.conv2d(
            result_19,
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
        input_177 = torch.nn.functional.batch_norm(
            input_176,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_176 = l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_178 = torch.nn.functional.silu(input_177, inplace=True)
        input_177 = None
        input_179 = torch.conv2d(
            input_178,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_178 = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_180 = torch.nn.functional.batch_norm(
            input_179,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_179 = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_181 = torch.nn.functional.silu(input_180, inplace=True)
        input_180 = None
        scale_70 = torch.nn.functional.adaptive_avg_pool2d(input_181, 1)
        scale_71 = torch.conv2d(
            scale_70,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_70 = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_72 = torch.nn.functional.silu(scale_71, inplace=True)
        scale_71 = None
        scale_73 = torch.conv2d(
            scale_72,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_72 = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_74 = torch.sigmoid(scale_73)
        scale_73 = None
        input_182 = scale_74 * input_181
        scale_74 = input_181 = None
        input_183 = torch.conv2d(
            input_182,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_182 = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_184 = torch.nn.functional.batch_norm(
            input_183,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_183 = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_20 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_20 = None
        input_184 += result_19
        result_20 = input_184
        input_184 = result_19 = None
        input_185 = torch.conv2d(
            result_20,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_20 = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_186 = torch.nn.functional.batch_norm(
            input_185,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_185 = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_187 = torch.nn.functional.silu(input_186, inplace=True)
        input_186 = None
        input_188 = torch.conv2d(
            input_187,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            960,
        )
        input_187 = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_189 = torch.nn.functional.batch_norm(
            input_188,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_188 = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_190 = torch.nn.functional.silu(input_189, inplace=True)
        input_189 = None
        scale_75 = torch.nn.functional.adaptive_avg_pool2d(input_190, 1)
        scale_76 = torch.conv2d(
            scale_75,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_75 = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_77 = torch.nn.functional.silu(scale_76, inplace=True)
        scale_76 = None
        scale_78 = torch.conv2d(
            scale_77,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_77 = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_79 = torch.sigmoid(scale_78)
        scale_78 = None
        input_191 = scale_79 * input_190
        scale_79 = input_190 = None
        input_192 = torch.conv2d(
            input_191,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_191 = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_193 = torch.nn.functional.batch_norm(
            input_192,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_192 = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_194 = torch.conv2d(
            input_193,
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
        input_195 = torch.nn.functional.batch_norm(
            input_194,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_194 = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_196 = torch.nn.functional.silu(input_195, inplace=True)
        input_195 = None
        input_197 = torch.conv2d(
            input_196,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        input_196 = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_198 = torch.nn.functional.batch_norm(
            input_197,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_197 = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_199 = torch.nn.functional.silu(input_198, inplace=True)
        input_198 = None
        scale_80 = torch.nn.functional.adaptive_avg_pool2d(input_199, 1)
        scale_81 = torch.conv2d(
            scale_80,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_80 = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_82 = torch.nn.functional.silu(scale_81, inplace=True)
        scale_81 = None
        scale_83 = torch.conv2d(
            scale_82,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_82 = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_84 = torch.sigmoid(scale_83)
        scale_83 = None
        input_200 = scale_84 * input_199
        scale_84 = input_199 = None
        input_201 = torch.conv2d(
            input_200,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_200 = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_202 = torch.nn.functional.batch_norm(
            input_201,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_201 = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_21 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_21 = None
        input_202 += input_193
        result_21 = input_202
        input_202 = input_193 = None
        input_203 = torch.conv2d(
            result_21,
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
        input_204 = torch.nn.functional.batch_norm(
            input_203,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_203 = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_205 = torch.nn.functional.silu(input_204, inplace=True)
        input_204 = None
        input_206 = torch.conv2d(
            input_205,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        input_205 = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_207 = torch.nn.functional.batch_norm(
            input_206,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_206 = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_208 = torch.nn.functional.silu(input_207, inplace=True)
        input_207 = None
        scale_85 = torch.nn.functional.adaptive_avg_pool2d(input_208, 1)
        scale_86 = torch.conv2d(
            scale_85,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_85 = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_87 = torch.nn.functional.silu(scale_86, inplace=True)
        scale_86 = None
        scale_88 = torch.conv2d(
            scale_87,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_87 = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_89 = torch.sigmoid(scale_88)
        scale_88 = None
        input_209 = scale_89 * input_208
        scale_89 = input_208 = None
        input_210 = torch.conv2d(
            input_209,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_209 = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_211 = torch.nn.functional.batch_norm(
            input_210,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_210 = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_22 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_22 = None
        input_211 += result_21
        result_22 = input_211
        input_211 = result_21 = None
        input_212 = torch.conv2d(
            result_22,
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
        input_213 = torch.nn.functional.batch_norm(
            input_212,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_212 = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_214 = torch.nn.functional.silu(input_213, inplace=True)
        input_213 = None
        input_215 = torch.conv2d(
            input_214,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        input_214 = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_216 = torch.nn.functional.batch_norm(
            input_215,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_215 = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_217 = torch.nn.functional.silu(input_216, inplace=True)
        input_216 = None
        scale_90 = torch.nn.functional.adaptive_avg_pool2d(input_217, 1)
        scale_91 = torch.conv2d(
            scale_90,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_90 = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_92 = torch.nn.functional.silu(scale_91, inplace=True)
        scale_91 = None
        scale_93 = torch.conv2d(
            scale_92,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_92 = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_94 = torch.sigmoid(scale_93)
        scale_93 = None
        input_218 = scale_94 * input_217
        scale_94 = input_217 = None
        input_219 = torch.conv2d(
            input_218,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_218 = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_220 = torch.nn.functional.batch_norm(
            input_219,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_219 = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_23 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_23 = None
        input_220 += result_22
        result_23 = input_220
        input_220 = result_22 = None
        input_221 = torch.conv2d(
            result_23,
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
        input_222 = torch.nn.functional.batch_norm(
            input_221,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_221 = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_223 = torch.nn.functional.silu(input_222, inplace=True)
        input_222 = None
        input_224 = torch.conv2d(
            input_223,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        input_223 = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_225 = torch.nn.functional.batch_norm(
            input_224,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_224 = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_226 = torch.nn.functional.silu(input_225, inplace=True)
        input_225 = None
        scale_95 = torch.nn.functional.adaptive_avg_pool2d(input_226, 1)
        scale_96 = torch.conv2d(
            scale_95,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_95 = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_97 = torch.nn.functional.silu(scale_96, inplace=True)
        scale_96 = None
        scale_98 = torch.conv2d(
            scale_97,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_97 = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_99 = torch.sigmoid(scale_98)
        scale_98 = None
        input_227 = scale_99 * input_226
        scale_99 = input_226 = None
        input_228 = torch.conv2d(
            input_227,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_227 = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_229 = torch.nn.functional.batch_norm(
            input_228,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_228 = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_24 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_24 = None
        input_229 += result_23
        result_24 = input_229
        input_229 = result_23 = None
        input_230 = torch.conv2d(
            result_24,
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
        input_231 = torch.nn.functional.batch_norm(
            input_230,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_230 = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_232 = torch.nn.functional.silu(input_231, inplace=True)
        input_231 = None
        input_233 = torch.conv2d(
            input_232,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        input_232 = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_234 = torch.nn.functional.batch_norm(
            input_233,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_233 = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_235 = torch.nn.functional.silu(input_234, inplace=True)
        input_234 = None
        scale_100 = torch.nn.functional.adaptive_avg_pool2d(input_235, 1)
        scale_101 = torch.conv2d(
            scale_100,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_100 = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_102 = torch.nn.functional.silu(scale_101, inplace=True)
        scale_101 = None
        scale_103 = torch.conv2d(
            scale_102,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_102 = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_104 = torch.sigmoid(scale_103)
        scale_103 = None
        input_236 = scale_104 * input_235
        scale_104 = input_235 = None
        input_237 = torch.conv2d(
            input_236,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_236 = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_238 = torch.nn.functional.batch_norm(
            input_237,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_237 = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_25 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_25 = None
        input_238 += result_24
        result_25 = input_238
        input_238 = result_24 = None
        input_239 = torch.conv2d(
            result_25,
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
        input_240 = torch.nn.functional.batch_norm(
            input_239,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_239 = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_241 = torch.nn.functional.silu(input_240, inplace=True)
        input_240 = None
        input_242 = torch.conv2d(
            input_241,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        input_241 = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_243 = torch.nn.functional.batch_norm(
            input_242,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_242 = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_244 = torch.nn.functional.silu(input_243, inplace=True)
        input_243 = None
        scale_105 = torch.nn.functional.adaptive_avg_pool2d(input_244, 1)
        scale_106 = torch.conv2d(
            scale_105,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_105 = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_107 = torch.nn.functional.silu(scale_106, inplace=True)
        scale_106 = None
        scale_108 = torch.conv2d(
            scale_107,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_107 = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_109 = torch.sigmoid(scale_108)
        scale_108 = None
        input_245 = scale_109 * input_244
        scale_109 = input_244 = None
        input_246 = torch.conv2d(
            input_245,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_245 = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_247 = torch.nn.functional.batch_norm(
            input_246,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_246 = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_26 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_26 = None
        input_247 += result_25
        result_26 = input_247
        input_247 = result_25 = None
        input_248 = torch.conv2d(
            result_26,
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
        input_249 = torch.nn.functional.batch_norm(
            input_248,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_248 = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_250 = torch.nn.functional.silu(input_249, inplace=True)
        input_249 = None
        input_251 = torch.conv2d(
            input_250,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        input_250 = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_252 = torch.nn.functional.batch_norm(
            input_251,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_251 = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_253 = torch.nn.functional.silu(input_252, inplace=True)
        input_252 = None
        scale_110 = torch.nn.functional.adaptive_avg_pool2d(input_253, 1)
        scale_111 = torch.conv2d(
            scale_110,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_110 = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_112 = torch.nn.functional.silu(scale_111, inplace=True)
        scale_111 = None
        scale_113 = torch.conv2d(
            scale_112,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_112 = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_114 = torch.sigmoid(scale_113)
        scale_113 = None
        input_254 = scale_114 * input_253
        scale_114 = input_253 = None
        input_255 = torch.conv2d(
            input_254,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_254 = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_256 = torch.nn.functional.batch_norm(
            input_255,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_255 = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_27 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_27 = None
        input_256 += result_26
        result_27 = input_256
        input_256 = result_26 = None
        input_257 = torch.conv2d(
            result_27,
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
        input_258 = torch.nn.functional.batch_norm(
            input_257,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_257 = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_259 = torch.nn.functional.silu(input_258, inplace=True)
        input_258 = None
        input_260 = torch.conv2d(
            input_259,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        input_259 = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_261 = torch.nn.functional.batch_norm(
            input_260,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_260 = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_262 = torch.nn.functional.silu(input_261, inplace=True)
        input_261 = None
        scale_115 = torch.nn.functional.adaptive_avg_pool2d(input_262, 1)
        scale_116 = torch.conv2d(
            scale_115,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_115 = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_117 = torch.nn.functional.silu(scale_116, inplace=True)
        scale_116 = None
        scale_118 = torch.conv2d(
            scale_117,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_117 = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_119 = torch.sigmoid(scale_118)
        scale_118 = None
        input_263 = scale_119 * input_262
        scale_119 = input_262 = None
        input_264 = torch.conv2d(
            input_263,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_263 = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_265 = torch.nn.functional.batch_norm(
            input_264,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_264 = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_28 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_28 = None
        input_265 += result_27
        result_28 = input_265
        input_265 = result_27 = None
        input_266 = torch.conv2d(
            result_28,
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
        input_267 = torch.nn.functional.batch_norm(
            input_266,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_266 = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_268 = torch.nn.functional.silu(input_267, inplace=True)
        input_267 = None
        input_269 = torch.conv2d(
            input_268,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        input_268 = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_270 = torch.nn.functional.batch_norm(
            input_269,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_269 = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_271 = torch.nn.functional.silu(input_270, inplace=True)
        input_270 = None
        scale_120 = torch.nn.functional.adaptive_avg_pool2d(input_271, 1)
        scale_121 = torch.conv2d(
            scale_120,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_120 = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_122 = torch.nn.functional.silu(scale_121, inplace=True)
        scale_121 = None
        scale_123 = torch.conv2d(
            scale_122,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_122 = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_124 = torch.sigmoid(scale_123)
        scale_123 = None
        input_272 = scale_124 * input_271
        scale_124 = input_271 = None
        input_273 = torch.conv2d(
            input_272,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_272 = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_274 = torch.nn.functional.batch_norm(
            input_273,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_273 = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_29 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_29 = None
        input_274 += result_28
        result_29 = input_274
        input_274 = result_28 = None
        input_275 = torch.conv2d(
            result_29,
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
        input_276 = torch.nn.functional.batch_norm(
            input_275,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_275 = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_277 = torch.nn.functional.silu(input_276, inplace=True)
        input_276 = None
        input_278 = torch.conv2d(
            input_277,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        input_277 = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_279 = torch.nn.functional.batch_norm(
            input_278,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_278 = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_280 = torch.nn.functional.silu(input_279, inplace=True)
        input_279 = None
        scale_125 = torch.nn.functional.adaptive_avg_pool2d(input_280, 1)
        scale_126 = torch.conv2d(
            scale_125,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_125 = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_127 = torch.nn.functional.silu(scale_126, inplace=True)
        scale_126 = None
        scale_128 = torch.conv2d(
            scale_127,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_127 = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_129 = torch.sigmoid(scale_128)
        scale_128 = None
        input_281 = scale_129 * input_280
        scale_129 = input_280 = None
        input_282 = torch.conv2d(
            input_281,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_281 = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_283 = torch.nn.functional.batch_norm(
            input_282,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_282 = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_30 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_30 = None
        input_283 += result_29
        result_30 = input_283
        input_283 = result_29 = None
        input_284 = torch.conv2d(
            result_30,
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
        input_285 = torch.nn.functional.batch_norm(
            input_284,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_284 = l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_286 = torch.nn.functional.silu(input_285, inplace=True)
        input_285 = None
        input_287 = torch.conv2d(
            input_286,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        input_286 = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_288 = torch.nn.functional.batch_norm(
            input_287,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_287 = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_289 = torch.nn.functional.silu(input_288, inplace=True)
        input_288 = None
        scale_130 = torch.nn.functional.adaptive_avg_pool2d(input_289, 1)
        scale_131 = torch.conv2d(
            scale_130,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_130 = l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_132 = torch.nn.functional.silu(scale_131, inplace=True)
        scale_131 = None
        scale_133 = torch.conv2d(
            scale_132,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_132 = l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_134 = torch.sigmoid(scale_133)
        scale_133 = None
        input_290 = scale_134 * input_289
        scale_134 = input_289 = None
        input_291 = torch.conv2d(
            input_290,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_290 = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_292 = torch.nn.functional.batch_norm(
            input_291,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_291 = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_31 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_31 = None
        input_292 += result_30
        result_31 = input_292
        input_292 = result_30 = None
        input_293 = torch.conv2d(
            result_31,
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
        input_294 = torch.nn.functional.batch_norm(
            input_293,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_293 = l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_295 = torch.nn.functional.silu(input_294, inplace=True)
        input_294 = None
        input_296 = torch.conv2d(
            input_295,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        input_295 = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_297 = torch.nn.functional.batch_norm(
            input_296,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_296 = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_298 = torch.nn.functional.silu(input_297, inplace=True)
        input_297 = None
        scale_135 = torch.nn.functional.adaptive_avg_pool2d(input_298, 1)
        scale_136 = torch.conv2d(
            scale_135,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_135 = l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_137 = torch.nn.functional.silu(scale_136, inplace=True)
        scale_136 = None
        scale_138 = torch.conv2d(
            scale_137,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_137 = l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_139 = torch.sigmoid(scale_138)
        scale_138 = None
        input_299 = scale_139 * input_298
        scale_139 = input_298 = None
        input_300 = torch.conv2d(
            input_299,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_299 = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_301 = torch.nn.functional.batch_norm(
            input_300,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_300 = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_32 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_32 = None
        input_301 += result_31
        result_32 = input_301
        input_301 = result_31 = None
        input_302 = torch.conv2d(
            result_32,
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
        input_303 = torch.nn.functional.batch_norm(
            input_302,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_302 = l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_304 = torch.nn.functional.silu(input_303, inplace=True)
        input_303 = None
        input_305 = torch.conv2d(
            input_304,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        input_304 = l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_306 = torch.nn.functional.batch_norm(
            input_305,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_305 = l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_307 = torch.nn.functional.silu(input_306, inplace=True)
        input_306 = None
        scale_140 = torch.nn.functional.adaptive_avg_pool2d(input_307, 1)
        scale_141 = torch.conv2d(
            scale_140,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_140 = l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_142 = torch.nn.functional.silu(scale_141, inplace=True)
        scale_141 = None
        scale_143 = torch.conv2d(
            scale_142,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_142 = l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_144 = torch.sigmoid(scale_143)
        scale_143 = None
        input_308 = scale_144 * input_307
        scale_144 = input_307 = None
        input_309 = torch.conv2d(
            input_308,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_308 = l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_310 = torch.nn.functional.batch_norm(
            input_309,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_309 = l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_13_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_33 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_33 = None
        input_310 += result_32
        result_33 = input_310
        input_310 = result_32 = None
        input_311 = torch.conv2d(
            result_33,
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
        input_312 = torch.nn.functional.batch_norm(
            input_311,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_311 = l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_313 = torch.nn.functional.silu(input_312, inplace=True)
        input_312 = None
        input_314 = torch.conv2d(
            input_313,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        input_313 = l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_315 = torch.nn.functional.batch_norm(
            input_314,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_314 = l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_316 = torch.nn.functional.silu(input_315, inplace=True)
        input_315 = None
        scale_145 = torch.nn.functional.adaptive_avg_pool2d(input_316, 1)
        scale_146 = torch.conv2d(
            scale_145,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_145 = l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_147 = torch.nn.functional.silu(scale_146, inplace=True)
        scale_146 = None
        scale_148 = torch.conv2d(
            scale_147,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_147 = l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_149 = torch.sigmoid(scale_148)
        scale_148 = None
        input_317 = scale_149 * input_316
        scale_149 = input_316 = None
        input_318 = torch.conv2d(
            input_317,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_317 = l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_319 = torch.nn.functional.batch_norm(
            input_318,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_318 = l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_14_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_34 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_34 = None
        input_319 += result_33
        result_34 = input_319
        input_319 = result_33 = None
        input_320 = torch.conv2d(
            result_34,
            l_self_modules_features_modules_7_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_34 = (
            l_self_modules_features_modules_7_modules_0_parameters_weight_
        ) = None
        input_321 = torch.nn.functional.batch_norm(
            input_320,
            l_self_modules_features_modules_7_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        input_320 = (
            l_self_modules_features_modules_7_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_7_modules_1_buffers_running_var_
        ) = (
            l_self_modules_features_modules_7_modules_1_parameters_weight_
        ) = l_self_modules_features_modules_7_modules_1_parameters_bias_ = None
        input_322 = torch.nn.functional.silu(input_321, inplace=True)
        input_321 = None
        x = torch.nn.functional.adaptive_avg_pool2d(input_322, 1)
        input_322 = None
        x_1 = torch.flatten(x, 1)
        x = None
        input_323 = torch.nn.functional.dropout(x_1, 0.2, False, True)
        x_1 = None
        input_324 = torch._C._nn.linear(
            input_323,
            l_self_modules_classifier_modules_1_parameters_weight_,
            l_self_modules_classifier_modules_1_parameters_bias_,
        )
        input_323 = (
            l_self_modules_classifier_modules_1_parameters_weight_
        ) = l_self_modules_classifier_modules_1_parameters_bias_ = None
        return (input_324,)
