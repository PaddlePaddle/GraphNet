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
        L_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc1_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc1_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc1_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc1_parameters_bias_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc2_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc2_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc2_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc2_parameters_bias_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_parameters_weight_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_parameters_weight_
        l_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_parameters_bias_ = L_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc1_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc1_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc1_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc1_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc2_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc2_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc2_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc2_parameters_bias_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_parameters_weight_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_parameters_weight_
        l_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_parameters_bias_ = L_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_parameters_bias_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc1_parameters_weight_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc1_parameters_weight_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc1_parameters_bias_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc1_parameters_bias_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc2_parameters_weight_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc2_parameters_weight_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc2_parameters_bias_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc2_parameters_bias_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_parameters_weight_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_parameters_weight_
        l_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_parameters_bias_ = L_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_parameters_bias_
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
        l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_parameters_bias_
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
        l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_parameters_bias_
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
        l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_parameters_bias_
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
        l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_parameters_bias_
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
        l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_parameters_bias_
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
        l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_parameters_bias_
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
        l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_parameters_bias_
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
        l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_parameters_bias_
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
        l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_parameters_bias_
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
        l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_parameters_bias_
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
        l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_parameters_bias_
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
        l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_parameters_bias_
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
            0.01,
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
            56,
        )
        input_3 = l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_4 = l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_6 = torch.nn.functional.silu(input_5, inplace=True)
        input_5 = None
        scale = torch.nn.functional.adaptive_avg_pool2d(input_6, 1)
        scale_1 = torch.conv2d(
            scale,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale = l_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc1_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc1_parameters_bias_ = (None)
        scale_2 = torch.nn.functional.silu(scale_1, inplace=True)
        scale_1 = None
        scale_3 = torch.conv2d(
            scale_2,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_2 = l_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc2_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_1_modules_fc2_parameters_bias_ = (None)
        scale_4 = torch.sigmoid(scale_3)
        scale_3 = None
        input_7 = scale_4 * input_6
        scale_4 = input_6 = None
        input_8 = torch.conv2d(
            input_7,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_7 = l_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_0_parameters_weight_ = (None)
        input_9 = torch.nn.functional.batch_norm(
            input_8,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_8 = l_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_buffers_running_var_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_parameters_weight_ = l_self_modules_features_modules_1_modules_0_modules_block_modules_2_modules_1_parameters_bias_ = (None)
        input_10 = torch.conv2d(
            input_9,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_11 = torch.nn.functional.batch_norm(
            input_10,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_10 = l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_12 = torch.nn.functional.silu(input_11, inplace=True)
        input_11 = None
        scale_5 = torch.nn.functional.adaptive_avg_pool2d(input_12, 1)
        scale_6 = torch.conv2d(
            scale_5,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_5 = l_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc1_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc1_parameters_bias_ = (None)
        scale_7 = torch.nn.functional.silu(scale_6, inplace=True)
        scale_6 = None
        scale_8 = torch.conv2d(
            scale_7,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_7 = l_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc2_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_1_modules_fc2_parameters_bias_ = (None)
        scale_9 = torch.sigmoid(scale_8)
        scale_8 = None
        input_13 = scale_9 * input_12
        scale_9 = input_12 = None
        input_14 = torch.conv2d(
            input_13,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_13 = l_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_0_parameters_weight_ = (None)
        input_15 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_14 = l_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_buffers_running_var_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_parameters_weight_ = l_self_modules_features_modules_1_modules_1_modules_block_modules_2_modules_1_parameters_bias_ = (None)
        _log_api_usage_once = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once = None
        input_15 += input_9
        result = input_15
        input_15 = input_9 = None
        input_16 = torch.conv2d(
            result,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_17 = torch.nn.functional.batch_norm(
            input_16,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_16 = l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_18 = torch.nn.functional.silu(input_17, inplace=True)
        input_17 = None
        scale_10 = torch.nn.functional.adaptive_avg_pool2d(input_18, 1)
        scale_11 = torch.conv2d(
            scale_10,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_10 = l_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc1_parameters_weight_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc1_parameters_bias_ = (None)
        scale_12 = torch.nn.functional.silu(scale_11, inplace=True)
        scale_11 = None
        scale_13 = torch.conv2d(
            scale_12,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_12 = l_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc2_parameters_weight_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_1_modules_fc2_parameters_bias_ = (None)
        scale_14 = torch.sigmoid(scale_13)
        scale_13 = None
        input_19 = scale_14 * input_18
        scale_14 = input_18 = None
        input_20 = torch.conv2d(
            input_19,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_19 = l_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_0_parameters_weight_ = (None)
        input_21 = torch.nn.functional.batch_norm(
            input_20,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_20 = l_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_buffers_running_var_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_parameters_weight_ = l_self_modules_features_modules_1_modules_2_modules_block_modules_2_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_1 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_1 = None
        input_21 += result
        result_1 = input_21
        input_21 = result = None
        input_22 = torch.conv2d(
            result_1,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_1 = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_23 = torch.nn.functional.batch_norm(
            input_22,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_22 = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_24 = torch.nn.functional.silu(input_23, inplace=True)
        input_23 = None
        input_25 = torch.conv2d(
            input_24,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            192,
        )
        input_24 = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_26 = torch.nn.functional.batch_norm(
            input_25,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_25 = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_27 = torch.nn.functional.silu(input_26, inplace=True)
        input_26 = None
        scale_15 = torch.nn.functional.adaptive_avg_pool2d(input_27, 1)
        scale_16 = torch.conv2d(
            scale_15,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_15 = l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_17 = torch.nn.functional.silu(scale_16, inplace=True)
        scale_16 = None
        scale_18 = torch.conv2d(
            scale_17,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_17 = l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_19 = torch.sigmoid(scale_18)
        scale_18 = None
        input_28 = scale_19 * input_27
        scale_19 = input_27 = None
        input_29 = torch.conv2d(
            input_28,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_28 = l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_30 = torch.nn.functional.batch_norm(
            input_29,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_29 = l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_31 = torch.conv2d(
            input_30,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_32 = torch.nn.functional.batch_norm(
            input_31,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_31 = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_33 = torch.nn.functional.silu(input_32, inplace=True)
        input_32 = None
        input_34 = torch.conv2d(
            input_33,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            240,
        )
        input_33 = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_35 = torch.nn.functional.batch_norm(
            input_34,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_34 = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_36 = torch.nn.functional.silu(input_35, inplace=True)
        input_35 = None
        scale_20 = torch.nn.functional.adaptive_avg_pool2d(input_36, 1)
        scale_21 = torch.conv2d(
            scale_20,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_20 = l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_22 = torch.nn.functional.silu(scale_21, inplace=True)
        scale_21 = None
        scale_23 = torch.conv2d(
            scale_22,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_22 = l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_24 = torch.sigmoid(scale_23)
        scale_23 = None
        input_37 = scale_24 * input_36
        scale_24 = input_36 = None
        input_38 = torch.conv2d(
            input_37,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_37 = l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_39 = torch.nn.functional.batch_norm(
            input_38,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_38 = l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_2 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_2 = None
        input_39 += input_30
        result_2 = input_39
        input_39 = input_30 = None
        input_40 = torch.conv2d(
            result_2,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_41 = torch.nn.functional.batch_norm(
            input_40,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_40 = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_42 = torch.nn.functional.silu(input_41, inplace=True)
        input_41 = None
        input_43 = torch.conv2d(
            input_42,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            240,
        )
        input_42 = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_44 = torch.nn.functional.batch_norm(
            input_43,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_43 = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_45 = torch.nn.functional.silu(input_44, inplace=True)
        input_44 = None
        scale_25 = torch.nn.functional.adaptive_avg_pool2d(input_45, 1)
        scale_26 = torch.conv2d(
            scale_25,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_25 = l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_27 = torch.nn.functional.silu(scale_26, inplace=True)
        scale_26 = None
        scale_28 = torch.conv2d(
            scale_27,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_27 = l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_29 = torch.sigmoid(scale_28)
        scale_28 = None
        input_46 = scale_29 * input_45
        scale_29 = input_45 = None
        input_47 = torch.conv2d(
            input_46,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_46 = l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_48 = torch.nn.functional.batch_norm(
            input_47,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_47 = l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_3 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_3 = None
        input_48 += result_2
        result_3 = input_48
        input_48 = result_2 = None
        input_49 = torch.conv2d(
            result_3,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_50 = torch.nn.functional.batch_norm(
            input_49,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_49 = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_51 = torch.nn.functional.silu(input_50, inplace=True)
        input_50 = None
        input_52 = torch.conv2d(
            input_51,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            240,
        )
        input_51 = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_53 = torch.nn.functional.batch_norm(
            input_52,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_52 = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_54 = torch.nn.functional.silu(input_53, inplace=True)
        input_53 = None
        scale_30 = torch.nn.functional.adaptive_avg_pool2d(input_54, 1)
        scale_31 = torch.conv2d(
            scale_30,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_30 = l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_32 = torch.nn.functional.silu(scale_31, inplace=True)
        scale_31 = None
        scale_33 = torch.conv2d(
            scale_32,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_32 = l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_34 = torch.sigmoid(scale_33)
        scale_33 = None
        input_55 = scale_34 * input_54
        scale_34 = input_54 = None
        input_56 = torch.conv2d(
            input_55,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_55 = l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_57 = torch.nn.functional.batch_norm(
            input_56,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_56 = l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_4 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_4 = None
        input_57 += result_3
        result_4 = input_57
        input_57 = result_3 = None
        input_58 = torch.conv2d(
            result_4,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_59 = torch.nn.functional.batch_norm(
            input_58,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_58 = l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_60 = torch.nn.functional.silu(input_59, inplace=True)
        input_59 = None
        input_61 = torch.conv2d(
            input_60,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            240,
        )
        input_60 = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_62 = torch.nn.functional.batch_norm(
            input_61,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_61 = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_63 = torch.nn.functional.silu(input_62, inplace=True)
        input_62 = None
        scale_35 = torch.nn.functional.adaptive_avg_pool2d(input_63, 1)
        scale_36 = torch.conv2d(
            scale_35,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_35 = l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_37 = torch.nn.functional.silu(scale_36, inplace=True)
        scale_36 = None
        scale_38 = torch.conv2d(
            scale_37,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_37 = l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_39 = torch.sigmoid(scale_38)
        scale_38 = None
        input_64 = scale_39 * input_63
        scale_39 = input_63 = None
        input_65 = torch.conv2d(
            input_64,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_64 = l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_66 = torch.nn.functional.batch_norm(
            input_65,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_65 = l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_5 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_5 = None
        input_66 += result_4
        result_5 = input_66
        input_66 = result_4 = None
        input_67 = torch.conv2d(
            result_5,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_68 = torch.nn.functional.batch_norm(
            input_67,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_67 = l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_69 = torch.nn.functional.silu(input_68, inplace=True)
        input_68 = None
        input_70 = torch.conv2d(
            input_69,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            240,
        )
        input_69 = l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_71 = torch.nn.functional.batch_norm(
            input_70,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_70 = l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_72 = torch.nn.functional.silu(input_71, inplace=True)
        input_71 = None
        scale_40 = torch.nn.functional.adaptive_avg_pool2d(input_72, 1)
        scale_41 = torch.conv2d(
            scale_40,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_40 = l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_42 = torch.nn.functional.silu(scale_41, inplace=True)
        scale_41 = None
        scale_43 = torch.conv2d(
            scale_42,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_42 = l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_44 = torch.sigmoid(scale_43)
        scale_43 = None
        input_73 = scale_44 * input_72
        scale_44 = input_72 = None
        input_74 = torch.conv2d(
            input_73,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_73 = l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_75 = torch.nn.functional.batch_norm(
            input_74,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_74 = l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_6 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_6 = None
        input_75 += result_5
        result_6 = input_75
        input_75 = result_5 = None
        input_76 = torch.conv2d(
            result_6,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_6 = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_77 = torch.nn.functional.batch_norm(
            input_76,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_76 = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_78 = torch.nn.functional.silu(input_77, inplace=True)
        input_77 = None
        input_79 = torch.conv2d(
            input_78,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            240,
        )
        input_78 = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_80 = torch.nn.functional.batch_norm(
            input_79,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_79 = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_81 = torch.nn.functional.silu(input_80, inplace=True)
        input_80 = None
        scale_45 = torch.nn.functional.adaptive_avg_pool2d(input_81, 1)
        scale_46 = torch.conv2d(
            scale_45,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_45 = l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_47 = torch.nn.functional.silu(scale_46, inplace=True)
        scale_46 = None
        scale_48 = torch.conv2d(
            scale_47,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_47 = l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_49 = torch.sigmoid(scale_48)
        scale_48 = None
        input_82 = scale_49 * input_81
        scale_49 = input_81 = None
        input_83 = torch.conv2d(
            input_82,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_82 = l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_84 = torch.nn.functional.batch_norm(
            input_83,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_83 = l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_85 = torch.conv2d(
            input_84,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_86 = torch.nn.functional.batch_norm(
            input_85,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_85 = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_87 = torch.nn.functional.silu(input_86, inplace=True)
        input_86 = None
        input_88 = torch.conv2d(
            input_87,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            432,
        )
        input_87 = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_89 = torch.nn.functional.batch_norm(
            input_88,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_88 = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_90 = torch.nn.functional.silu(input_89, inplace=True)
        input_89 = None
        scale_50 = torch.nn.functional.adaptive_avg_pool2d(input_90, 1)
        scale_51 = torch.conv2d(
            scale_50,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_50 = l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_52 = torch.nn.functional.silu(scale_51, inplace=True)
        scale_51 = None
        scale_53 = torch.conv2d(
            scale_52,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_52 = l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_54 = torch.sigmoid(scale_53)
        scale_53 = None
        input_91 = scale_54 * input_90
        scale_54 = input_90 = None
        input_92 = torch.conv2d(
            input_91,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_91 = l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_93 = torch.nn.functional.batch_norm(
            input_92,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_92 = l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_7 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_7 = None
        input_93 += input_84
        result_7 = input_93
        input_93 = input_84 = None
        input_94 = torch.conv2d(
            result_7,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_95 = torch.nn.functional.batch_norm(
            input_94,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_94 = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_96 = torch.nn.functional.silu(input_95, inplace=True)
        input_95 = None
        input_97 = torch.conv2d(
            input_96,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            432,
        )
        input_96 = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_98 = torch.nn.functional.batch_norm(
            input_97,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_97 = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_99 = torch.nn.functional.silu(input_98, inplace=True)
        input_98 = None
        scale_55 = torch.nn.functional.adaptive_avg_pool2d(input_99, 1)
        scale_56 = torch.conv2d(
            scale_55,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_55 = l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_57 = torch.nn.functional.silu(scale_56, inplace=True)
        scale_56 = None
        scale_58 = torch.conv2d(
            scale_57,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_57 = l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_59 = torch.sigmoid(scale_58)
        scale_58 = None
        input_100 = scale_59 * input_99
        scale_59 = input_99 = None
        input_101 = torch.conv2d(
            input_100,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_100 = l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_102 = torch.nn.functional.batch_norm(
            input_101,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_101 = l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_8 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_8 = None
        input_102 += result_7
        result_8 = input_102
        input_102 = result_7 = None
        input_103 = torch.conv2d(
            result_8,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_104 = torch.nn.functional.batch_norm(
            input_103,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_103 = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_105 = torch.nn.functional.silu(input_104, inplace=True)
        input_104 = None
        input_106 = torch.conv2d(
            input_105,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            432,
        )
        input_105 = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_107 = torch.nn.functional.batch_norm(
            input_106,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_106 = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_108 = torch.nn.functional.silu(input_107, inplace=True)
        input_107 = None
        scale_60 = torch.nn.functional.adaptive_avg_pool2d(input_108, 1)
        scale_61 = torch.conv2d(
            scale_60,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_60 = l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_62 = torch.nn.functional.silu(scale_61, inplace=True)
        scale_61 = None
        scale_63 = torch.conv2d(
            scale_62,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_62 = l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_64 = torch.sigmoid(scale_63)
        scale_63 = None
        input_109 = scale_64 * input_108
        scale_64 = input_108 = None
        input_110 = torch.conv2d(
            input_109,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_109 = l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_111 = torch.nn.functional.batch_norm(
            input_110,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_110 = l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_9 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_9 = None
        input_111 += result_8
        result_9 = input_111
        input_111 = result_8 = None
        input_112 = torch.conv2d(
            result_9,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_113 = torch.nn.functional.batch_norm(
            input_112,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_112 = l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_114 = torch.nn.functional.silu(input_113, inplace=True)
        input_113 = None
        input_115 = torch.conv2d(
            input_114,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            432,
        )
        input_114 = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_116 = torch.nn.functional.batch_norm(
            input_115,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_115 = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_117 = torch.nn.functional.silu(input_116, inplace=True)
        input_116 = None
        scale_65 = torch.nn.functional.adaptive_avg_pool2d(input_117, 1)
        scale_66 = torch.conv2d(
            scale_65,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_65 = l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_67 = torch.nn.functional.silu(scale_66, inplace=True)
        scale_66 = None
        scale_68 = torch.conv2d(
            scale_67,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_67 = l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_69 = torch.sigmoid(scale_68)
        scale_68 = None
        input_118 = scale_69 * input_117
        scale_69 = input_117 = None
        input_119 = torch.conv2d(
            input_118,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_118 = l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_120 = torch.nn.functional.batch_norm(
            input_119,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_119 = l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_10 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_10 = None
        input_120 += result_9
        result_10 = input_120
        input_120 = result_9 = None
        input_121 = torch.conv2d(
            result_10,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_122 = torch.nn.functional.batch_norm(
            input_121,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_121 = l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_123 = torch.nn.functional.silu(input_122, inplace=True)
        input_122 = None
        input_124 = torch.conv2d(
            input_123,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            432,
        )
        input_123 = l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_125 = torch.nn.functional.batch_norm(
            input_124,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_124 = l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_126 = torch.nn.functional.silu(input_125, inplace=True)
        input_125 = None
        scale_70 = torch.nn.functional.adaptive_avg_pool2d(input_126, 1)
        scale_71 = torch.conv2d(
            scale_70,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_70 = l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_72 = torch.nn.functional.silu(scale_71, inplace=True)
        scale_71 = None
        scale_73 = torch.conv2d(
            scale_72,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_72 = l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_74 = torch.sigmoid(scale_73)
        scale_73 = None
        input_127 = scale_74 * input_126
        scale_74 = input_126 = None
        input_128 = torch.conv2d(
            input_127,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_127 = l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_129 = torch.nn.functional.batch_norm(
            input_128,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_128 = l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_11 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_11 = None
        input_129 += result_10
        result_11 = input_129
        input_129 = result_10 = None
        input_130 = torch.conv2d(
            result_11,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_11 = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_131 = torch.nn.functional.batch_norm(
            input_130,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_130 = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_132 = torch.nn.functional.silu(input_131, inplace=True)
        input_131 = None
        input_133 = torch.conv2d(
            input_132,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            432,
        )
        input_132 = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_134 = torch.nn.functional.batch_norm(
            input_133,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_133 = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_135 = torch.nn.functional.silu(input_134, inplace=True)
        input_134 = None
        scale_75 = torch.nn.functional.adaptive_avg_pool2d(input_135, 1)
        scale_76 = torch.conv2d(
            scale_75,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_75 = l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_77 = torch.nn.functional.silu(scale_76, inplace=True)
        scale_76 = None
        scale_78 = torch.conv2d(
            scale_77,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_77 = l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_79 = torch.sigmoid(scale_78)
        scale_78 = None
        input_136 = scale_79 * input_135
        scale_79 = input_135 = None
        input_137 = torch.conv2d(
            input_136,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_136 = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_138 = torch.nn.functional.batch_norm(
            input_137,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_137 = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_139 = torch.conv2d(
            input_138,
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
        input_140 = torch.nn.functional.batch_norm(
            input_139,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_139 = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_141 = torch.nn.functional.silu(input_140, inplace=True)
        input_140 = None
        input_142 = torch.conv2d(
            input_141,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            864,
        )
        input_141 = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_143 = torch.nn.functional.batch_norm(
            input_142,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_142 = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_144 = torch.nn.functional.silu(input_143, inplace=True)
        input_143 = None
        scale_80 = torch.nn.functional.adaptive_avg_pool2d(input_144, 1)
        scale_81 = torch.conv2d(
            scale_80,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_80 = l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_82 = torch.nn.functional.silu(scale_81, inplace=True)
        scale_81 = None
        scale_83 = torch.conv2d(
            scale_82,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_82 = l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_84 = torch.sigmoid(scale_83)
        scale_83 = None
        input_145 = scale_84 * input_144
        scale_84 = input_144 = None
        input_146 = torch.conv2d(
            input_145,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_145 = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_147 = torch.nn.functional.batch_norm(
            input_146,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_146 = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_12 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_12 = None
        input_147 += input_138
        result_12 = input_147
        input_147 = input_138 = None
        input_148 = torch.conv2d(
            result_12,
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
        input_149 = torch.nn.functional.batch_norm(
            input_148,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_148 = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_150 = torch.nn.functional.silu(input_149, inplace=True)
        input_149 = None
        input_151 = torch.conv2d(
            input_150,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            864,
        )
        input_150 = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_152 = torch.nn.functional.batch_norm(
            input_151,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_151 = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_153 = torch.nn.functional.silu(input_152, inplace=True)
        input_152 = None
        scale_85 = torch.nn.functional.adaptive_avg_pool2d(input_153, 1)
        scale_86 = torch.conv2d(
            scale_85,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_85 = l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_87 = torch.nn.functional.silu(scale_86, inplace=True)
        scale_86 = None
        scale_88 = torch.conv2d(
            scale_87,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_87 = l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_89 = torch.sigmoid(scale_88)
        scale_88 = None
        input_154 = scale_89 * input_153
        scale_89 = input_153 = None
        input_155 = torch.conv2d(
            input_154,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_154 = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_156 = torch.nn.functional.batch_norm(
            input_155,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_155 = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_13 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_13 = None
        input_156 += result_12
        result_13 = input_156
        input_156 = result_12 = None
        input_157 = torch.conv2d(
            result_13,
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
        input_158 = torch.nn.functional.batch_norm(
            input_157,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_157 = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_159 = torch.nn.functional.silu(input_158, inplace=True)
        input_158 = None
        input_160 = torch.conv2d(
            input_159,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            864,
        )
        input_159 = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_161 = torch.nn.functional.batch_norm(
            input_160,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_160 = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_162 = torch.nn.functional.silu(input_161, inplace=True)
        input_161 = None
        scale_90 = torch.nn.functional.adaptive_avg_pool2d(input_162, 1)
        scale_91 = torch.conv2d(
            scale_90,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_90 = l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_92 = torch.nn.functional.silu(scale_91, inplace=True)
        scale_91 = None
        scale_93 = torch.conv2d(
            scale_92,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_92 = l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_94 = torch.sigmoid(scale_93)
        scale_93 = None
        input_163 = scale_94 * input_162
        scale_94 = input_162 = None
        input_164 = torch.conv2d(
            input_163,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_163 = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_165 = torch.nn.functional.batch_norm(
            input_164,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_164 = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_14 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_14 = None
        input_165 += result_13
        result_14 = input_165
        input_165 = result_13 = None
        input_166 = torch.conv2d(
            result_14,
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
        input_167 = torch.nn.functional.batch_norm(
            input_166,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_166 = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_168 = torch.nn.functional.silu(input_167, inplace=True)
        input_167 = None
        input_169 = torch.conv2d(
            input_168,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            864,
        )
        input_168 = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_170 = torch.nn.functional.batch_norm(
            input_169,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_169 = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_171 = torch.nn.functional.silu(input_170, inplace=True)
        input_170 = None
        scale_95 = torch.nn.functional.adaptive_avg_pool2d(input_171, 1)
        scale_96 = torch.conv2d(
            scale_95,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_95 = l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_97 = torch.nn.functional.silu(scale_96, inplace=True)
        scale_96 = None
        scale_98 = torch.conv2d(
            scale_97,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_97 = l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_99 = torch.sigmoid(scale_98)
        scale_98 = None
        input_172 = scale_99 * input_171
        scale_99 = input_171 = None
        input_173 = torch.conv2d(
            input_172,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_172 = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_174 = torch.nn.functional.batch_norm(
            input_173,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_173 = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_15 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_15 = None
        input_174 += result_14
        result_15 = input_174
        input_174 = result_14 = None
        input_175 = torch.conv2d(
            result_15,
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
        input_176 = torch.nn.functional.batch_norm(
            input_175,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_175 = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_177 = torch.nn.functional.silu(input_176, inplace=True)
        input_176 = None
        input_178 = torch.conv2d(
            input_177,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            864,
        )
        input_177 = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_179 = torch.nn.functional.batch_norm(
            input_178,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_178 = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_180 = torch.nn.functional.silu(input_179, inplace=True)
        input_179 = None
        scale_100 = torch.nn.functional.adaptive_avg_pool2d(input_180, 1)
        scale_101 = torch.conv2d(
            scale_100,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_100 = l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_102 = torch.nn.functional.silu(scale_101, inplace=True)
        scale_101 = None
        scale_103 = torch.conv2d(
            scale_102,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_102 = l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_104 = torch.sigmoid(scale_103)
        scale_103 = None
        input_181 = scale_104 * input_180
        scale_104 = input_180 = None
        input_182 = torch.conv2d(
            input_181,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_181 = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_183 = torch.nn.functional.batch_norm(
            input_182,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_182 = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_16 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_16 = None
        input_183 += result_15
        result_16 = input_183
        input_183 = result_15 = None
        input_184 = torch.conv2d(
            result_16,
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
        input_185 = torch.nn.functional.batch_norm(
            input_184,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_184 = l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_186 = torch.nn.functional.silu(input_185, inplace=True)
        input_185 = None
        input_187 = torch.conv2d(
            input_186,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            864,
        )
        input_186 = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_188 = torch.nn.functional.batch_norm(
            input_187,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_187 = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_189 = torch.nn.functional.silu(input_188, inplace=True)
        input_188 = None
        scale_105 = torch.nn.functional.adaptive_avg_pool2d(input_189, 1)
        scale_106 = torch.conv2d(
            scale_105,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_105 = l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_107 = torch.nn.functional.silu(scale_106, inplace=True)
        scale_106 = None
        scale_108 = torch.conv2d(
            scale_107,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_107 = l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_109 = torch.sigmoid(scale_108)
        scale_108 = None
        input_190 = scale_109 * input_189
        scale_109 = input_189 = None
        input_191 = torch.conv2d(
            input_190,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_190 = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_192 = torch.nn.functional.batch_norm(
            input_191,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_191 = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_17 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_17 = None
        input_192 += result_16
        result_17 = input_192
        input_192 = result_16 = None
        input_193 = torch.conv2d(
            result_17,
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
        input_194 = torch.nn.functional.batch_norm(
            input_193,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_193 = l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_195 = torch.nn.functional.silu(input_194, inplace=True)
        input_194 = None
        input_196 = torch.conv2d(
            input_195,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            864,
        )
        input_195 = l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_197 = torch.nn.functional.batch_norm(
            input_196,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_196 = l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_198 = torch.nn.functional.silu(input_197, inplace=True)
        input_197 = None
        scale_110 = torch.nn.functional.adaptive_avg_pool2d(input_198, 1)
        scale_111 = torch.conv2d(
            scale_110,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_110 = l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_112 = torch.nn.functional.silu(scale_111, inplace=True)
        scale_111 = None
        scale_113 = torch.conv2d(
            scale_112,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_112 = l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_114 = torch.sigmoid(scale_113)
        scale_113 = None
        input_199 = scale_114 * input_198
        scale_114 = input_198 = None
        input_200 = torch.conv2d(
            input_199,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_199 = l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_201 = torch.nn.functional.batch_norm(
            input_200,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_200 = l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_18 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_18 = None
        input_201 += result_17
        result_18 = input_201
        input_201 = result_17 = None
        input_202 = torch.conv2d(
            result_18,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_18 = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_203 = torch.nn.functional.batch_norm(
            input_202,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_202 = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_204 = torch.nn.functional.silu(input_203, inplace=True)
        input_203 = None
        input_205 = torch.conv2d(
            input_204,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            864,
        )
        input_204 = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_206 = torch.nn.functional.batch_norm(
            input_205,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_205 = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_207 = torch.nn.functional.silu(input_206, inplace=True)
        input_206 = None
        scale_115 = torch.nn.functional.adaptive_avg_pool2d(input_207, 1)
        scale_116 = torch.conv2d(
            scale_115,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_115 = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_117 = torch.nn.functional.silu(scale_116, inplace=True)
        scale_116 = None
        scale_118 = torch.conv2d(
            scale_117,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_117 = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_119 = torch.sigmoid(scale_118)
        scale_118 = None
        input_208 = scale_119 * input_207
        scale_119 = input_207 = None
        input_209 = torch.conv2d(
            input_208,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_208 = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_210 = torch.nn.functional.batch_norm(
            input_209,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_209 = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_211 = torch.conv2d(
            input_210,
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
        input_212 = torch.nn.functional.batch_norm(
            input_211,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_211 = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_213 = torch.nn.functional.silu(input_212, inplace=True)
        input_212 = None
        input_214 = torch.conv2d(
            input_213,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1200,
        )
        input_213 = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_215 = torch.nn.functional.batch_norm(
            input_214,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_214 = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_216 = torch.nn.functional.silu(input_215, inplace=True)
        input_215 = None
        scale_120 = torch.nn.functional.adaptive_avg_pool2d(input_216, 1)
        scale_121 = torch.conv2d(
            scale_120,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_120 = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_122 = torch.nn.functional.silu(scale_121, inplace=True)
        scale_121 = None
        scale_123 = torch.conv2d(
            scale_122,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_122 = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_124 = torch.sigmoid(scale_123)
        scale_123 = None
        input_217 = scale_124 * input_216
        scale_124 = input_216 = None
        input_218 = torch.conv2d(
            input_217,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_217 = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_219 = torch.nn.functional.batch_norm(
            input_218,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_218 = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_19 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_19 = None
        input_219 += input_210
        result_19 = input_219
        input_219 = input_210 = None
        input_220 = torch.conv2d(
            result_19,
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
        input_221 = torch.nn.functional.batch_norm(
            input_220,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_220 = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_222 = torch.nn.functional.silu(input_221, inplace=True)
        input_221 = None
        input_223 = torch.conv2d(
            input_222,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1200,
        )
        input_222 = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_224 = torch.nn.functional.batch_norm(
            input_223,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_223 = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_225 = torch.nn.functional.silu(input_224, inplace=True)
        input_224 = None
        scale_125 = torch.nn.functional.adaptive_avg_pool2d(input_225, 1)
        scale_126 = torch.conv2d(
            scale_125,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_125 = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_127 = torch.nn.functional.silu(scale_126, inplace=True)
        scale_126 = None
        scale_128 = torch.conv2d(
            scale_127,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_127 = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_129 = torch.sigmoid(scale_128)
        scale_128 = None
        input_226 = scale_129 * input_225
        scale_129 = input_225 = None
        input_227 = torch.conv2d(
            input_226,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_226 = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_228 = torch.nn.functional.batch_norm(
            input_227,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_227 = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_20 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_20 = None
        input_228 += result_19
        result_20 = input_228
        input_228 = result_19 = None
        input_229 = torch.conv2d(
            result_20,
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
        input_230 = torch.nn.functional.batch_norm(
            input_229,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_229 = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_231 = torch.nn.functional.silu(input_230, inplace=True)
        input_230 = None
        input_232 = torch.conv2d(
            input_231,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1200,
        )
        input_231 = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_233 = torch.nn.functional.batch_norm(
            input_232,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_232 = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_234 = torch.nn.functional.silu(input_233, inplace=True)
        input_233 = None
        scale_130 = torch.nn.functional.adaptive_avg_pool2d(input_234, 1)
        scale_131 = torch.conv2d(
            scale_130,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_130 = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_132 = torch.nn.functional.silu(scale_131, inplace=True)
        scale_131 = None
        scale_133 = torch.conv2d(
            scale_132,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_132 = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_134 = torch.sigmoid(scale_133)
        scale_133 = None
        input_235 = scale_134 * input_234
        scale_134 = input_234 = None
        input_236 = torch.conv2d(
            input_235,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_235 = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_237 = torch.nn.functional.batch_norm(
            input_236,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_236 = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_21 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_21 = None
        input_237 += result_20
        result_21 = input_237
        input_237 = result_20 = None
        input_238 = torch.conv2d(
            result_21,
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
        input_239 = torch.nn.functional.batch_norm(
            input_238,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_238 = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_240 = torch.nn.functional.silu(input_239, inplace=True)
        input_239 = None
        input_241 = torch.conv2d(
            input_240,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1200,
        )
        input_240 = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_242 = torch.nn.functional.batch_norm(
            input_241,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_241 = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_243 = torch.nn.functional.silu(input_242, inplace=True)
        input_242 = None
        scale_135 = torch.nn.functional.adaptive_avg_pool2d(input_243, 1)
        scale_136 = torch.conv2d(
            scale_135,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_135 = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_137 = torch.nn.functional.silu(scale_136, inplace=True)
        scale_136 = None
        scale_138 = torch.conv2d(
            scale_137,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_137 = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_139 = torch.sigmoid(scale_138)
        scale_138 = None
        input_244 = scale_139 * input_243
        scale_139 = input_243 = None
        input_245 = torch.conv2d(
            input_244,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_244 = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_246 = torch.nn.functional.batch_norm(
            input_245,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_245 = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_22 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_22 = None
        input_246 += result_21
        result_22 = input_246
        input_246 = result_21 = None
        input_247 = torch.conv2d(
            result_22,
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
        input_248 = torch.nn.functional.batch_norm(
            input_247,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_247 = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_249 = torch.nn.functional.silu(input_248, inplace=True)
        input_248 = None
        input_250 = torch.conv2d(
            input_249,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1200,
        )
        input_249 = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_251 = torch.nn.functional.batch_norm(
            input_250,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_250 = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_252 = torch.nn.functional.silu(input_251, inplace=True)
        input_251 = None
        scale_140 = torch.nn.functional.adaptive_avg_pool2d(input_252, 1)
        scale_141 = torch.conv2d(
            scale_140,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_140 = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_142 = torch.nn.functional.silu(scale_141, inplace=True)
        scale_141 = None
        scale_143 = torch.conv2d(
            scale_142,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_142 = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_144 = torch.sigmoid(scale_143)
        scale_143 = None
        input_253 = scale_144 * input_252
        scale_144 = input_252 = None
        input_254 = torch.conv2d(
            input_253,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_253 = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_255 = torch.nn.functional.batch_norm(
            input_254,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_254 = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_23 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_23 = None
        input_255 += result_22
        result_23 = input_255
        input_255 = result_22 = None
        input_256 = torch.conv2d(
            result_23,
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
        input_257 = torch.nn.functional.batch_norm(
            input_256,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_256 = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_258 = torch.nn.functional.silu(input_257, inplace=True)
        input_257 = None
        input_259 = torch.conv2d(
            input_258,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1200,
        )
        input_258 = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_260 = torch.nn.functional.batch_norm(
            input_259,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_259 = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_261 = torch.nn.functional.silu(input_260, inplace=True)
        input_260 = None
        scale_145 = torch.nn.functional.adaptive_avg_pool2d(input_261, 1)
        scale_146 = torch.conv2d(
            scale_145,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_145 = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_147 = torch.nn.functional.silu(scale_146, inplace=True)
        scale_146 = None
        scale_148 = torch.conv2d(
            scale_147,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_147 = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_149 = torch.sigmoid(scale_148)
        scale_148 = None
        input_262 = scale_149 * input_261
        scale_149 = input_261 = None
        input_263 = torch.conv2d(
            input_262,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_262 = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_264 = torch.nn.functional.batch_norm(
            input_263,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_263 = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_24 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_24 = None
        input_264 += result_23
        result_24 = input_264
        input_264 = result_23 = None
        input_265 = torch.conv2d(
            result_24,
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
        input_266 = torch.nn.functional.batch_norm(
            input_265,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_265 = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_267 = torch.nn.functional.silu(input_266, inplace=True)
        input_266 = None
        input_268 = torch.conv2d(
            input_267,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1200,
        )
        input_267 = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_269 = torch.nn.functional.batch_norm(
            input_268,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_268 = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_270 = torch.nn.functional.silu(input_269, inplace=True)
        input_269 = None
        scale_150 = torch.nn.functional.adaptive_avg_pool2d(input_270, 1)
        scale_151 = torch.conv2d(
            scale_150,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_150 = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_152 = torch.nn.functional.silu(scale_151, inplace=True)
        scale_151 = None
        scale_153 = torch.conv2d(
            scale_152,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_152 = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_154 = torch.sigmoid(scale_153)
        scale_153 = None
        input_271 = scale_154 * input_270
        scale_154 = input_270 = None
        input_272 = torch.conv2d(
            input_271,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_271 = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_273 = torch.nn.functional.batch_norm(
            input_272,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_272 = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_25 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_25 = None
        input_273 += result_24
        result_25 = input_273
        input_273 = result_24 = None
        input_274 = torch.conv2d(
            result_25,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_25 = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_275 = torch.nn.functional.batch_norm(
            input_274,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_274 = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_276 = torch.nn.functional.silu(input_275, inplace=True)
        input_275 = None
        input_277 = torch.conv2d(
            input_276,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            1200,
        )
        input_276 = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_278 = torch.nn.functional.batch_norm(
            input_277,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_277 = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_279 = torch.nn.functional.silu(input_278, inplace=True)
        input_278 = None
        scale_155 = torch.nn.functional.adaptive_avg_pool2d(input_279, 1)
        scale_156 = torch.conv2d(
            scale_155,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_155 = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_157 = torch.nn.functional.silu(scale_156, inplace=True)
        scale_156 = None
        scale_158 = torch.conv2d(
            scale_157,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_157 = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_159 = torch.sigmoid(scale_158)
        scale_158 = None
        input_280 = scale_159 * input_279
        scale_159 = input_279 = None
        input_281 = torch.conv2d(
            input_280,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_280 = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_282 = torch.nn.functional.batch_norm(
            input_281,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_281 = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_283 = torch.conv2d(
            input_282,
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
        input_284 = torch.nn.functional.batch_norm(
            input_283,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_283 = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_285 = torch.nn.functional.silu(input_284, inplace=True)
        input_284 = None
        input_286 = torch.conv2d(
            input_285,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2064,
        )
        input_285 = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_287 = torch.nn.functional.batch_norm(
            input_286,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_286 = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_288 = torch.nn.functional.silu(input_287, inplace=True)
        input_287 = None
        scale_160 = torch.nn.functional.adaptive_avg_pool2d(input_288, 1)
        scale_161 = torch.conv2d(
            scale_160,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_160 = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_162 = torch.nn.functional.silu(scale_161, inplace=True)
        scale_161 = None
        scale_163 = torch.conv2d(
            scale_162,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_162 = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_164 = torch.sigmoid(scale_163)
        scale_163 = None
        input_289 = scale_164 * input_288
        scale_164 = input_288 = None
        input_290 = torch.conv2d(
            input_289,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_289 = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_291 = torch.nn.functional.batch_norm(
            input_290,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_290 = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_26 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_26 = None
        input_291 += input_282
        result_26 = input_291
        input_291 = input_282 = None
        input_292 = torch.conv2d(
            result_26,
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
        input_293 = torch.nn.functional.batch_norm(
            input_292,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_292 = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_294 = torch.nn.functional.silu(input_293, inplace=True)
        input_293 = None
        input_295 = torch.conv2d(
            input_294,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2064,
        )
        input_294 = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_296 = torch.nn.functional.batch_norm(
            input_295,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_295 = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_297 = torch.nn.functional.silu(input_296, inplace=True)
        input_296 = None
        scale_165 = torch.nn.functional.adaptive_avg_pool2d(input_297, 1)
        scale_166 = torch.conv2d(
            scale_165,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_165 = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_167 = torch.nn.functional.silu(scale_166, inplace=True)
        scale_166 = None
        scale_168 = torch.conv2d(
            scale_167,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_167 = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_169 = torch.sigmoid(scale_168)
        scale_168 = None
        input_298 = scale_169 * input_297
        scale_169 = input_297 = None
        input_299 = torch.conv2d(
            input_298,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_298 = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_300 = torch.nn.functional.batch_norm(
            input_299,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_299 = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_27 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_27 = None
        input_300 += result_26
        result_27 = input_300
        input_300 = result_26 = None
        input_301 = torch.conv2d(
            result_27,
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
        input_302 = torch.nn.functional.batch_norm(
            input_301,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_301 = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_303 = torch.nn.functional.silu(input_302, inplace=True)
        input_302 = None
        input_304 = torch.conv2d(
            input_303,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2064,
        )
        input_303 = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_305 = torch.nn.functional.batch_norm(
            input_304,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_304 = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_306 = torch.nn.functional.silu(input_305, inplace=True)
        input_305 = None
        scale_170 = torch.nn.functional.adaptive_avg_pool2d(input_306, 1)
        scale_171 = torch.conv2d(
            scale_170,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_170 = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_172 = torch.nn.functional.silu(scale_171, inplace=True)
        scale_171 = None
        scale_173 = torch.conv2d(
            scale_172,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_172 = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_174 = torch.sigmoid(scale_173)
        scale_173 = None
        input_307 = scale_174 * input_306
        scale_174 = input_306 = None
        input_308 = torch.conv2d(
            input_307,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_307 = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_309 = torch.nn.functional.batch_norm(
            input_308,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_308 = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_28 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_28 = None
        input_309 += result_27
        result_28 = input_309
        input_309 = result_27 = None
        input_310 = torch.conv2d(
            result_28,
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
        input_311 = torch.nn.functional.batch_norm(
            input_310,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_310 = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_312 = torch.nn.functional.silu(input_311, inplace=True)
        input_311 = None
        input_313 = torch.conv2d(
            input_312,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2064,
        )
        input_312 = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_314 = torch.nn.functional.batch_norm(
            input_313,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_313 = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_315 = torch.nn.functional.silu(input_314, inplace=True)
        input_314 = None
        scale_175 = torch.nn.functional.adaptive_avg_pool2d(input_315, 1)
        scale_176 = torch.conv2d(
            scale_175,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_175 = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_177 = torch.nn.functional.silu(scale_176, inplace=True)
        scale_176 = None
        scale_178 = torch.conv2d(
            scale_177,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_177 = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_179 = torch.sigmoid(scale_178)
        scale_178 = None
        input_316 = scale_179 * input_315
        scale_179 = input_315 = None
        input_317 = torch.conv2d(
            input_316,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_316 = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_318 = torch.nn.functional.batch_norm(
            input_317,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_317 = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_29 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_29 = None
        input_318 += result_28
        result_29 = input_318
        input_318 = result_28 = None
        input_319 = torch.conv2d(
            result_29,
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
        input_320 = torch.nn.functional.batch_norm(
            input_319,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_319 = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_321 = torch.nn.functional.silu(input_320, inplace=True)
        input_320 = None
        input_322 = torch.conv2d(
            input_321,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2064,
        )
        input_321 = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_323 = torch.nn.functional.batch_norm(
            input_322,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_322 = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_324 = torch.nn.functional.silu(input_323, inplace=True)
        input_323 = None
        scale_180 = torch.nn.functional.adaptive_avg_pool2d(input_324, 1)
        scale_181 = torch.conv2d(
            scale_180,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_180 = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_182 = torch.nn.functional.silu(scale_181, inplace=True)
        scale_181 = None
        scale_183 = torch.conv2d(
            scale_182,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_182 = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_184 = torch.sigmoid(scale_183)
        scale_183 = None
        input_325 = scale_184 * input_324
        scale_184 = input_324 = None
        input_326 = torch.conv2d(
            input_325,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_325 = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_327 = torch.nn.functional.batch_norm(
            input_326,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_326 = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_30 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_30 = None
        input_327 += result_29
        result_30 = input_327
        input_327 = result_29 = None
        input_328 = torch.conv2d(
            result_30,
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
        input_329 = torch.nn.functional.batch_norm(
            input_328,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_328 = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_330 = torch.nn.functional.silu(input_329, inplace=True)
        input_329 = None
        input_331 = torch.conv2d(
            input_330,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2064,
        )
        input_330 = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_332 = torch.nn.functional.batch_norm(
            input_331,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_331 = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_333 = torch.nn.functional.silu(input_332, inplace=True)
        input_332 = None
        scale_185 = torch.nn.functional.adaptive_avg_pool2d(input_333, 1)
        scale_186 = torch.conv2d(
            scale_185,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_185 = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_187 = torch.nn.functional.silu(scale_186, inplace=True)
        scale_186 = None
        scale_188 = torch.conv2d(
            scale_187,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_187 = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_189 = torch.sigmoid(scale_188)
        scale_188 = None
        input_334 = scale_189 * input_333
        scale_189 = input_333 = None
        input_335 = torch.conv2d(
            input_334,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_334 = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_336 = torch.nn.functional.batch_norm(
            input_335,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_335 = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_31 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_31 = None
        input_336 += result_30
        result_31 = input_336
        input_336 = result_30 = None
        input_337 = torch.conv2d(
            result_31,
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
        input_338 = torch.nn.functional.batch_norm(
            input_337,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_337 = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_339 = torch.nn.functional.silu(input_338, inplace=True)
        input_338 = None
        input_340 = torch.conv2d(
            input_339,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2064,
        )
        input_339 = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_341 = torch.nn.functional.batch_norm(
            input_340,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_340 = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_342 = torch.nn.functional.silu(input_341, inplace=True)
        input_341 = None
        scale_190 = torch.nn.functional.adaptive_avg_pool2d(input_342, 1)
        scale_191 = torch.conv2d(
            scale_190,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_190 = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_192 = torch.nn.functional.silu(scale_191, inplace=True)
        scale_191 = None
        scale_193 = torch.conv2d(
            scale_192,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_192 = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_194 = torch.sigmoid(scale_193)
        scale_193 = None
        input_343 = scale_194 * input_342
        scale_194 = input_342 = None
        input_344 = torch.conv2d(
            input_343,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_343 = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_345 = torch.nn.functional.batch_norm(
            input_344,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_344 = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_32 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_32 = None
        input_345 += result_31
        result_32 = input_345
        input_345 = result_31 = None
        input_346 = torch.conv2d(
            result_32,
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
        input_347 = torch.nn.functional.batch_norm(
            input_346,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_346 = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_348 = torch.nn.functional.silu(input_347, inplace=True)
        input_347 = None
        input_349 = torch.conv2d(
            input_348,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2064,
        )
        input_348 = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_350 = torch.nn.functional.batch_norm(
            input_349,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_349 = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_351 = torch.nn.functional.silu(input_350, inplace=True)
        input_350 = None
        scale_195 = torch.nn.functional.adaptive_avg_pool2d(input_351, 1)
        scale_196 = torch.conv2d(
            scale_195,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_195 = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_197 = torch.nn.functional.silu(scale_196, inplace=True)
        scale_196 = None
        scale_198 = torch.conv2d(
            scale_197,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_197 = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_199 = torch.sigmoid(scale_198)
        scale_198 = None
        input_352 = scale_199 * input_351
        scale_199 = input_351 = None
        input_353 = torch.conv2d(
            input_352,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_352 = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_354 = torch.nn.functional.batch_norm(
            input_353,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_353 = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_33 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_33 = None
        input_354 += result_32
        result_33 = input_354
        input_354 = result_32 = None
        input_355 = torch.conv2d(
            result_33,
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
        input_356 = torch.nn.functional.batch_norm(
            input_355,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_355 = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_357 = torch.nn.functional.silu(input_356, inplace=True)
        input_356 = None
        input_358 = torch.conv2d(
            input_357,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2064,
        )
        input_357 = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_359 = torch.nn.functional.batch_norm(
            input_358,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_358 = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_360 = torch.nn.functional.silu(input_359, inplace=True)
        input_359 = None
        scale_200 = torch.nn.functional.adaptive_avg_pool2d(input_360, 1)
        scale_201 = torch.conv2d(
            scale_200,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_200 = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_202 = torch.nn.functional.silu(scale_201, inplace=True)
        scale_201 = None
        scale_203 = torch.conv2d(
            scale_202,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_202 = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_204 = torch.sigmoid(scale_203)
        scale_203 = None
        input_361 = scale_204 * input_360
        scale_204 = input_360 = None
        input_362 = torch.conv2d(
            input_361,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_361 = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_363 = torch.nn.functional.batch_norm(
            input_362,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_362 = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_34 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_34 = None
        input_363 += result_33
        result_34 = input_363
        input_363 = result_33 = None
        input_364 = torch.conv2d(
            result_34,
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
        input_365 = torch.nn.functional.batch_norm(
            input_364,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_364 = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_366 = torch.nn.functional.silu(input_365, inplace=True)
        input_365 = None
        input_367 = torch.conv2d(
            input_366,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2064,
        )
        input_366 = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_368 = torch.nn.functional.batch_norm(
            input_367,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_367 = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_369 = torch.nn.functional.silu(input_368, inplace=True)
        input_368 = None
        scale_205 = torch.nn.functional.adaptive_avg_pool2d(input_369, 1)
        scale_206 = torch.conv2d(
            scale_205,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_205 = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_207 = torch.nn.functional.silu(scale_206, inplace=True)
        scale_206 = None
        scale_208 = torch.conv2d(
            scale_207,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_207 = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_209 = torch.sigmoid(scale_208)
        scale_208 = None
        input_370 = scale_209 * input_369
        scale_209 = input_369 = None
        input_371 = torch.conv2d(
            input_370,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_370 = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_372 = torch.nn.functional.batch_norm(
            input_371,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_371 = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_35 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_35 = None
        input_372 += result_34
        result_35 = input_372
        input_372 = result_34 = None
        input_373 = torch.conv2d(
            result_35,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_35 = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_374 = torch.nn.functional.batch_norm(
            input_373,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_373 = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_375 = torch.nn.functional.silu(input_374, inplace=True)
        input_374 = None
        input_376 = torch.conv2d(
            input_375,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2064,
        )
        input_375 = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_377 = torch.nn.functional.batch_norm(
            input_376,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_376 = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_378 = torch.nn.functional.silu(input_377, inplace=True)
        input_377 = None
        scale_210 = torch.nn.functional.adaptive_avg_pool2d(input_378, 1)
        scale_211 = torch.conv2d(
            scale_210,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_210 = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_212 = torch.nn.functional.silu(scale_211, inplace=True)
        scale_211 = None
        scale_213 = torch.conv2d(
            scale_212,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_212 = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_214 = torch.sigmoid(scale_213)
        scale_213 = None
        input_379 = scale_214 * input_378
        scale_214 = input_378 = None
        input_380 = torch.conv2d(
            input_379,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_379 = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_381 = torch.nn.functional.batch_norm(
            input_380,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_380 = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_382 = torch.conv2d(
            input_381,
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
        input_383 = torch.nn.functional.batch_norm(
            input_382,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_382 = l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_384 = torch.nn.functional.silu(input_383, inplace=True)
        input_383 = None
        input_385 = torch.conv2d(
            input_384,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3456,
        )
        input_384 = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_386 = torch.nn.functional.batch_norm(
            input_385,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_385 = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_387 = torch.nn.functional.silu(input_386, inplace=True)
        input_386 = None
        scale_215 = torch.nn.functional.adaptive_avg_pool2d(input_387, 1)
        scale_216 = torch.conv2d(
            scale_215,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_215 = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_217 = torch.nn.functional.silu(scale_216, inplace=True)
        scale_216 = None
        scale_218 = torch.conv2d(
            scale_217,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_217 = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_219 = torch.sigmoid(scale_218)
        scale_218 = None
        input_388 = scale_219 * input_387
        scale_219 = input_387 = None
        input_389 = torch.conv2d(
            input_388,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_388 = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_390 = torch.nn.functional.batch_norm(
            input_389,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_389 = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_36 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_36 = None
        input_390 += input_381
        result_36 = input_390
        input_390 = input_381 = None
        input_391 = torch.conv2d(
            result_36,
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
        input_392 = torch.nn.functional.batch_norm(
            input_391,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_391 = l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_393 = torch.nn.functional.silu(input_392, inplace=True)
        input_392 = None
        input_394 = torch.conv2d(
            input_393,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3456,
        )
        input_393 = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_395 = torch.nn.functional.batch_norm(
            input_394,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_394 = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_396 = torch.nn.functional.silu(input_395, inplace=True)
        input_395 = None
        scale_220 = torch.nn.functional.adaptive_avg_pool2d(input_396, 1)
        scale_221 = torch.conv2d(
            scale_220,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_220 = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_222 = torch.nn.functional.silu(scale_221, inplace=True)
        scale_221 = None
        scale_223 = torch.conv2d(
            scale_222,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_222 = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_224 = torch.sigmoid(scale_223)
        scale_223 = None
        input_397 = scale_224 * input_396
        scale_224 = input_396 = None
        input_398 = torch.conv2d(
            input_397,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_397 = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_399 = torch.nn.functional.batch_norm(
            input_398,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_398 = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_37 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_37 = None
        input_399 += result_36
        result_37 = input_399
        input_399 = result_36 = None
        input_400 = torch.conv2d(
            result_37,
            l_self_modules_features_modules_8_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_37 = (
            l_self_modules_features_modules_8_modules_0_parameters_weight_
        ) = None
        input_401 = torch.nn.functional.batch_norm(
            input_400,
            l_self_modules_features_modules_8_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_1_buffers_running_var_,
            l_self_modules_features_modules_8_modules_1_parameters_weight_,
            l_self_modules_features_modules_8_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_400 = (
            l_self_modules_features_modules_8_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_8_modules_1_buffers_running_var_
        ) = (
            l_self_modules_features_modules_8_modules_1_parameters_weight_
        ) = l_self_modules_features_modules_8_modules_1_parameters_bias_ = None
        input_402 = torch.nn.functional.silu(input_401, inplace=True)
        input_401 = None
        x = torch.nn.functional.adaptive_avg_pool2d(input_402, 1)
        input_402 = None
        x_1 = torch.flatten(x, 1)
        x = None
        input_403 = torch.nn.functional.dropout(x_1, 0.5, False, True)
        x_1 = None
        input_404 = torch._C._nn.linear(
            input_403,
            l_self_modules_classifier_modules_1_parameters_weight_,
            l_self_modules_classifier_modules_1_parameters_bias_,
        )
        input_403 = (
            l_self_modules_classifier_modules_1_parameters_weight_
        ) = l_self_modules_classifier_modules_1_parameters_bias_ = None
        return (input_404,)
