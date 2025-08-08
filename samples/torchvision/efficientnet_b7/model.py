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
        L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc1_parameters_weight_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc1_parameters_weight_
        l_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc1_parameters_bias_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc1_parameters_bias_
        l_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc2_parameters_weight_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc2_parameters_weight_
        l_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc2_parameters_bias_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc2_parameters_bias_
        l_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_parameters_weight_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_parameters_weight_
        l_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_parameters_bias_ = L_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_parameters_bias_
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
        l_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_parameters_bias_
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
        l_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_parameters_bias_
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
            64,
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
            l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_23 = torch.nn.functional.batch_norm(
            input_22,
            l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_22 = l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_1_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_24 = torch.nn.functional.silu(input_23, inplace=True)
        input_23 = None
        scale_15 = torch.nn.functional.adaptive_avg_pool2d(input_24, 1)
        scale_16 = torch.conv2d(
            scale_15,
            l_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_15 = l_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc1_parameters_weight_ = l_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc1_parameters_bias_ = (None)
        scale_17 = torch.nn.functional.silu(scale_16, inplace=True)
        scale_16 = None
        scale_18 = torch.conv2d(
            scale_17,
            l_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_17 = l_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc2_parameters_weight_ = l_self_modules_features_modules_1_modules_3_modules_block_modules_1_modules_fc2_parameters_bias_ = (None)
        scale_19 = torch.sigmoid(scale_18)
        scale_18 = None
        input_25 = scale_19 * input_24
        scale_19 = input_24 = None
        input_26 = torch.conv2d(
            input_25,
            l_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_25 = l_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_0_parameters_weight_ = (None)
        input_27 = torch.nn.functional.batch_norm(
            input_26,
            l_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_26 = l_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_buffers_running_var_ = l_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_parameters_weight_ = l_self_modules_features_modules_1_modules_3_modules_block_modules_2_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_2 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_2 = None
        input_27 += result_1
        result_2 = input_27
        input_27 = result_1 = None
        input_28 = torch.conv2d(
            result_2,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_2 = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_29 = torch.nn.functional.batch_norm(
            input_28,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_28 = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_30 = torch.nn.functional.silu(input_29, inplace=True)
        input_29 = None
        input_31 = torch.conv2d(
            input_30,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            192,
        )
        input_30 = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_32 = torch.nn.functional.batch_norm(
            input_31,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_31 = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_33 = torch.nn.functional.silu(input_32, inplace=True)
        input_32 = None
        scale_20 = torch.nn.functional.adaptive_avg_pool2d(input_33, 1)
        scale_21 = torch.conv2d(
            scale_20,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_20 = l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_22 = torch.nn.functional.silu(scale_21, inplace=True)
        scale_21 = None
        scale_23 = torch.conv2d(
            scale_22,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_22 = l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_24 = torch.sigmoid(scale_23)
        scale_23 = None
        input_34 = scale_24 * input_33
        scale_24 = input_33 = None
        input_35 = torch.conv2d(
            input_34,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_34 = l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_36 = torch.nn.functional.batch_norm(
            input_35,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_35 = l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_37 = torch.conv2d(
            input_36,
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
        input_38 = torch.nn.functional.batch_norm(
            input_37,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_37 = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_39 = torch.nn.functional.silu(input_38, inplace=True)
        input_38 = None
        input_40 = torch.conv2d(
            input_39,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            288,
        )
        input_39 = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_41 = torch.nn.functional.batch_norm(
            input_40,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_40 = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_42 = torch.nn.functional.silu(input_41, inplace=True)
        input_41 = None
        scale_25 = torch.nn.functional.adaptive_avg_pool2d(input_42, 1)
        scale_26 = torch.conv2d(
            scale_25,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_25 = l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_27 = torch.nn.functional.silu(scale_26, inplace=True)
        scale_26 = None
        scale_28 = torch.conv2d(
            scale_27,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_27 = l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_29 = torch.sigmoid(scale_28)
        scale_28 = None
        input_43 = scale_29 * input_42
        scale_29 = input_42 = None
        input_44 = torch.conv2d(
            input_43,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_43 = l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_45 = torch.nn.functional.batch_norm(
            input_44,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_44 = l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_3 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_3 = None
        input_45 += input_36
        result_3 = input_45
        input_45 = input_36 = None
        input_46 = torch.conv2d(
            result_3,
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
        input_47 = torch.nn.functional.batch_norm(
            input_46,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_46 = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_48 = torch.nn.functional.silu(input_47, inplace=True)
        input_47 = None
        input_49 = torch.conv2d(
            input_48,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            288,
        )
        input_48 = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_50 = torch.nn.functional.batch_norm(
            input_49,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_49 = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_51 = torch.nn.functional.silu(input_50, inplace=True)
        input_50 = None
        scale_30 = torch.nn.functional.adaptive_avg_pool2d(input_51, 1)
        scale_31 = torch.conv2d(
            scale_30,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_30 = l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_32 = torch.nn.functional.silu(scale_31, inplace=True)
        scale_31 = None
        scale_33 = torch.conv2d(
            scale_32,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_32 = l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_34 = torch.sigmoid(scale_33)
        scale_33 = None
        input_52 = scale_34 * input_51
        scale_34 = input_51 = None
        input_53 = torch.conv2d(
            input_52,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_52 = l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_54 = torch.nn.functional.batch_norm(
            input_53,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_53 = l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_4 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_4 = None
        input_54 += result_3
        result_4 = input_54
        input_54 = result_3 = None
        input_55 = torch.conv2d(
            result_4,
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
        input_56 = torch.nn.functional.batch_norm(
            input_55,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_55 = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_57 = torch.nn.functional.silu(input_56, inplace=True)
        input_56 = None
        input_58 = torch.conv2d(
            input_57,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            288,
        )
        input_57 = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_59 = torch.nn.functional.batch_norm(
            input_58,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_58 = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_60 = torch.nn.functional.silu(input_59, inplace=True)
        input_59 = None
        scale_35 = torch.nn.functional.adaptive_avg_pool2d(input_60, 1)
        scale_36 = torch.conv2d(
            scale_35,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_35 = l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_37 = torch.nn.functional.silu(scale_36, inplace=True)
        scale_36 = None
        scale_38 = torch.conv2d(
            scale_37,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_37 = l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_39 = torch.sigmoid(scale_38)
        scale_38 = None
        input_61 = scale_39 * input_60
        scale_39 = input_60 = None
        input_62 = torch.conv2d(
            input_61,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_61 = l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_63 = torch.nn.functional.batch_norm(
            input_62,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_62 = l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_5 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_5 = None
        input_63 += result_4
        result_5 = input_63
        input_63 = result_4 = None
        input_64 = torch.conv2d(
            result_5,
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
        input_65 = torch.nn.functional.batch_norm(
            input_64,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_64 = l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_66 = torch.nn.functional.silu(input_65, inplace=True)
        input_65 = None
        input_67 = torch.conv2d(
            input_66,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            288,
        )
        input_66 = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_68 = torch.nn.functional.batch_norm(
            input_67,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_67 = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_69 = torch.nn.functional.silu(input_68, inplace=True)
        input_68 = None
        scale_40 = torch.nn.functional.adaptive_avg_pool2d(input_69, 1)
        scale_41 = torch.conv2d(
            scale_40,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_40 = l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_42 = torch.nn.functional.silu(scale_41, inplace=True)
        scale_41 = None
        scale_43 = torch.conv2d(
            scale_42,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_42 = l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_44 = torch.sigmoid(scale_43)
        scale_43 = None
        input_70 = scale_44 * input_69
        scale_44 = input_69 = None
        input_71 = torch.conv2d(
            input_70,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_70 = l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_72 = torch.nn.functional.batch_norm(
            input_71,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_71 = l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_6 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_6 = None
        input_72 += result_5
        result_6 = input_72
        input_72 = result_5 = None
        input_73 = torch.conv2d(
            result_6,
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
        input_74 = torch.nn.functional.batch_norm(
            input_73,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_73 = l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_75 = torch.nn.functional.silu(input_74, inplace=True)
        input_74 = None
        input_76 = torch.conv2d(
            input_75,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            288,
        )
        input_75 = l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_77 = torch.nn.functional.batch_norm(
            input_76,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_76 = l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_78 = torch.nn.functional.silu(input_77, inplace=True)
        input_77 = None
        scale_45 = torch.nn.functional.adaptive_avg_pool2d(input_78, 1)
        scale_46 = torch.conv2d(
            scale_45,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_45 = l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_47 = torch.nn.functional.silu(scale_46, inplace=True)
        scale_46 = None
        scale_48 = torch.conv2d(
            scale_47,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_47 = l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_49 = torch.sigmoid(scale_48)
        scale_48 = None
        input_79 = scale_49 * input_78
        scale_49 = input_78 = None
        input_80 = torch.conv2d(
            input_79,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_79 = l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_81 = torch.nn.functional.batch_norm(
            input_80,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_80 = l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_7 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_7 = None
        input_81 += result_6
        result_7 = input_81
        input_81 = result_6 = None
        input_82 = torch.conv2d(
            result_7,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_83 = torch.nn.functional.batch_norm(
            input_82,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_82 = l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_84 = torch.nn.functional.silu(input_83, inplace=True)
        input_83 = None
        input_85 = torch.conv2d(
            input_84,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            288,
        )
        input_84 = l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_86 = torch.nn.functional.batch_norm(
            input_85,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_85 = l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_87 = torch.nn.functional.silu(input_86, inplace=True)
        input_86 = None
        scale_50 = torch.nn.functional.adaptive_avg_pool2d(input_87, 1)
        scale_51 = torch.conv2d(
            scale_50,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_50 = l_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_52 = torch.nn.functional.silu(scale_51, inplace=True)
        scale_51 = None
        scale_53 = torch.conv2d(
            scale_52,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_52 = l_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_2_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_54 = torch.sigmoid(scale_53)
        scale_53 = None
        input_88 = scale_54 * input_87
        scale_54 = input_87 = None
        input_89 = torch.conv2d(
            input_88,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_88 = l_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_90 = torch.nn.functional.batch_norm(
            input_89,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_89 = l_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_8 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_8 = None
        input_90 += result_7
        result_8 = input_90
        input_90 = result_7 = None
        input_91 = torch.conv2d(
            result_8,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_8 = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_92 = torch.nn.functional.batch_norm(
            input_91,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_91 = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_93 = torch.nn.functional.silu(input_92, inplace=True)
        input_92 = None
        input_94 = torch.conv2d(
            input_93,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            288,
        )
        input_93 = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_95 = torch.nn.functional.batch_norm(
            input_94,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_94 = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_96 = torch.nn.functional.silu(input_95, inplace=True)
        input_95 = None
        scale_55 = torch.nn.functional.adaptive_avg_pool2d(input_96, 1)
        scale_56 = torch.conv2d(
            scale_55,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_55 = l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_57 = torch.nn.functional.silu(scale_56, inplace=True)
        scale_56 = None
        scale_58 = torch.conv2d(
            scale_57,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_57 = l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_59 = torch.sigmoid(scale_58)
        scale_58 = None
        input_97 = scale_59 * input_96
        scale_59 = input_96 = None
        input_98 = torch.conv2d(
            input_97,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_97 = l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_99 = torch.nn.functional.batch_norm(
            input_98,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_98 = l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_100 = torch.conv2d(
            input_99,
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
        input_101 = torch.nn.functional.batch_norm(
            input_100,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_100 = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_102 = torch.nn.functional.silu(input_101, inplace=True)
        input_101 = None
        input_103 = torch.conv2d(
            input_102,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            480,
        )
        input_102 = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_104 = torch.nn.functional.batch_norm(
            input_103,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_103 = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_105 = torch.nn.functional.silu(input_104, inplace=True)
        input_104 = None
        scale_60 = torch.nn.functional.adaptive_avg_pool2d(input_105, 1)
        scale_61 = torch.conv2d(
            scale_60,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_60 = l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_62 = torch.nn.functional.silu(scale_61, inplace=True)
        scale_61 = None
        scale_63 = torch.conv2d(
            scale_62,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_62 = l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_64 = torch.sigmoid(scale_63)
        scale_63 = None
        input_106 = scale_64 * input_105
        scale_64 = input_105 = None
        input_107 = torch.conv2d(
            input_106,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_106 = l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_108 = torch.nn.functional.batch_norm(
            input_107,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_107 = l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_9 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_9 = None
        input_108 += input_99
        result_9 = input_108
        input_108 = input_99 = None
        input_109 = torch.conv2d(
            result_9,
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
        input_110 = torch.nn.functional.batch_norm(
            input_109,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_109 = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_111 = torch.nn.functional.silu(input_110, inplace=True)
        input_110 = None
        input_112 = torch.conv2d(
            input_111,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            480,
        )
        input_111 = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_113 = torch.nn.functional.batch_norm(
            input_112,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_112 = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_114 = torch.nn.functional.silu(input_113, inplace=True)
        input_113 = None
        scale_65 = torch.nn.functional.adaptive_avg_pool2d(input_114, 1)
        scale_66 = torch.conv2d(
            scale_65,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_65 = l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_67 = torch.nn.functional.silu(scale_66, inplace=True)
        scale_66 = None
        scale_68 = torch.conv2d(
            scale_67,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_67 = l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_69 = torch.sigmoid(scale_68)
        scale_68 = None
        input_115 = scale_69 * input_114
        scale_69 = input_114 = None
        input_116 = torch.conv2d(
            input_115,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_115 = l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_117 = torch.nn.functional.batch_norm(
            input_116,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_116 = l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_10 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_10 = None
        input_117 += result_9
        result_10 = input_117
        input_117 = result_9 = None
        input_118 = torch.conv2d(
            result_10,
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
        input_119 = torch.nn.functional.batch_norm(
            input_118,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_118 = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_120 = torch.nn.functional.silu(input_119, inplace=True)
        input_119 = None
        input_121 = torch.conv2d(
            input_120,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            480,
        )
        input_120 = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_122 = torch.nn.functional.batch_norm(
            input_121,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_121 = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_123 = torch.nn.functional.silu(input_122, inplace=True)
        input_122 = None
        scale_70 = torch.nn.functional.adaptive_avg_pool2d(input_123, 1)
        scale_71 = torch.conv2d(
            scale_70,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_70 = l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_72 = torch.nn.functional.silu(scale_71, inplace=True)
        scale_71 = None
        scale_73 = torch.conv2d(
            scale_72,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_72 = l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_74 = torch.sigmoid(scale_73)
        scale_73 = None
        input_124 = scale_74 * input_123
        scale_74 = input_123 = None
        input_125 = torch.conv2d(
            input_124,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_124 = l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_126 = torch.nn.functional.batch_norm(
            input_125,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_125 = l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_11 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_11 = None
        input_126 += result_10
        result_11 = input_126
        input_126 = result_10 = None
        input_127 = torch.conv2d(
            result_11,
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
        input_128 = torch.nn.functional.batch_norm(
            input_127,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_127 = l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_129 = torch.nn.functional.silu(input_128, inplace=True)
        input_128 = None
        input_130 = torch.conv2d(
            input_129,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            480,
        )
        input_129 = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_131 = torch.nn.functional.batch_norm(
            input_130,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_130 = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_132 = torch.nn.functional.silu(input_131, inplace=True)
        input_131 = None
        scale_75 = torch.nn.functional.adaptive_avg_pool2d(input_132, 1)
        scale_76 = torch.conv2d(
            scale_75,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_75 = l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_77 = torch.nn.functional.silu(scale_76, inplace=True)
        scale_76 = None
        scale_78 = torch.conv2d(
            scale_77,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_77 = l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_79 = torch.sigmoid(scale_78)
        scale_78 = None
        input_133 = scale_79 * input_132
        scale_79 = input_132 = None
        input_134 = torch.conv2d(
            input_133,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_133 = l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_135 = torch.nn.functional.batch_norm(
            input_134,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_134 = l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_12 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_12 = None
        input_135 += result_11
        result_12 = input_135
        input_135 = result_11 = None
        input_136 = torch.conv2d(
            result_12,
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
        input_137 = torch.nn.functional.batch_norm(
            input_136,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_136 = l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_138 = torch.nn.functional.silu(input_137, inplace=True)
        input_137 = None
        input_139 = torch.conv2d(
            input_138,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            480,
        )
        input_138 = l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_140 = torch.nn.functional.batch_norm(
            input_139,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_139 = l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_141 = torch.nn.functional.silu(input_140, inplace=True)
        input_140 = None
        scale_80 = torch.nn.functional.adaptive_avg_pool2d(input_141, 1)
        scale_81 = torch.conv2d(
            scale_80,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_80 = l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_82 = torch.nn.functional.silu(scale_81, inplace=True)
        scale_81 = None
        scale_83 = torch.conv2d(
            scale_82,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_82 = l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_84 = torch.sigmoid(scale_83)
        scale_83 = None
        input_142 = scale_84 * input_141
        scale_84 = input_141 = None
        input_143 = torch.conv2d(
            input_142,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_142 = l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_144 = torch.nn.functional.batch_norm(
            input_143,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_143 = l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_13 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_13 = None
        input_144 += result_12
        result_13 = input_144
        input_144 = result_12 = None
        input_145 = torch.conv2d(
            result_13,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_146 = torch.nn.functional.batch_norm(
            input_145,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_145 = l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_147 = torch.nn.functional.silu(input_146, inplace=True)
        input_146 = None
        input_148 = torch.conv2d(
            input_147,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            480,
        )
        input_147 = l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_149 = torch.nn.functional.batch_norm(
            input_148,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_148 = l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_150 = torch.nn.functional.silu(input_149, inplace=True)
        input_149 = None
        scale_85 = torch.nn.functional.adaptive_avg_pool2d(input_150, 1)
        scale_86 = torch.conv2d(
            scale_85,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_85 = l_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_87 = torch.nn.functional.silu(scale_86, inplace=True)
        scale_86 = None
        scale_88 = torch.conv2d(
            scale_87,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_87 = l_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_3_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_89 = torch.sigmoid(scale_88)
        scale_88 = None
        input_151 = scale_89 * input_150
        scale_89 = input_150 = None
        input_152 = torch.conv2d(
            input_151,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_151 = l_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_153 = torch.nn.functional.batch_norm(
            input_152,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_152 = l_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_14 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_14 = None
        input_153 += result_13
        result_14 = input_153
        input_153 = result_13 = None
        input_154 = torch.conv2d(
            result_14,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_14 = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_155 = torch.nn.functional.batch_norm(
            input_154,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_154 = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_156 = torch.nn.functional.silu(input_155, inplace=True)
        input_155 = None
        input_157 = torch.conv2d(
            input_156,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            480,
        )
        input_156 = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_158 = torch.nn.functional.batch_norm(
            input_157,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_157 = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_159 = torch.nn.functional.silu(input_158, inplace=True)
        input_158 = None
        scale_90 = torch.nn.functional.adaptive_avg_pool2d(input_159, 1)
        scale_91 = torch.conv2d(
            scale_90,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_90 = l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_92 = torch.nn.functional.silu(scale_91, inplace=True)
        scale_91 = None
        scale_93 = torch.conv2d(
            scale_92,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_92 = l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_94 = torch.sigmoid(scale_93)
        scale_93 = None
        input_160 = scale_94 * input_159
        scale_94 = input_159 = None
        input_161 = torch.conv2d(
            input_160,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_160 = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_162 = torch.nn.functional.batch_norm(
            input_161,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_161 = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_163 = torch.conv2d(
            input_162,
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
        input_164 = torch.nn.functional.batch_norm(
            input_163,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_163 = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_165 = torch.nn.functional.silu(input_164, inplace=True)
        input_164 = None
        input_166 = torch.conv2d(
            input_165,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_165 = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_167 = torch.nn.functional.batch_norm(
            input_166,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_166 = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_168 = torch.nn.functional.silu(input_167, inplace=True)
        input_167 = None
        scale_95 = torch.nn.functional.adaptive_avg_pool2d(input_168, 1)
        scale_96 = torch.conv2d(
            scale_95,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_95 = l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_97 = torch.nn.functional.silu(scale_96, inplace=True)
        scale_96 = None
        scale_98 = torch.conv2d(
            scale_97,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_97 = l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_99 = torch.sigmoid(scale_98)
        scale_98 = None
        input_169 = scale_99 * input_168
        scale_99 = input_168 = None
        input_170 = torch.conv2d(
            input_169,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_169 = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_171 = torch.nn.functional.batch_norm(
            input_170,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_170 = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_15 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_15 = None
        input_171 += input_162
        result_15 = input_171
        input_171 = input_162 = None
        input_172 = torch.conv2d(
            result_15,
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
        input_173 = torch.nn.functional.batch_norm(
            input_172,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_172 = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_174 = torch.nn.functional.silu(input_173, inplace=True)
        input_173 = None
        input_175 = torch.conv2d(
            input_174,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_174 = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_176 = torch.nn.functional.batch_norm(
            input_175,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_175 = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_177 = torch.nn.functional.silu(input_176, inplace=True)
        input_176 = None
        scale_100 = torch.nn.functional.adaptive_avg_pool2d(input_177, 1)
        scale_101 = torch.conv2d(
            scale_100,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_100 = l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_102 = torch.nn.functional.silu(scale_101, inplace=True)
        scale_101 = None
        scale_103 = torch.conv2d(
            scale_102,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_102 = l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_104 = torch.sigmoid(scale_103)
        scale_103 = None
        input_178 = scale_104 * input_177
        scale_104 = input_177 = None
        input_179 = torch.conv2d(
            input_178,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_178 = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_180 = torch.nn.functional.batch_norm(
            input_179,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_179 = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_16 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_16 = None
        input_180 += result_15
        result_16 = input_180
        input_180 = result_15 = None
        input_181 = torch.conv2d(
            result_16,
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
        input_182 = torch.nn.functional.batch_norm(
            input_181,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_181 = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_183 = torch.nn.functional.silu(input_182, inplace=True)
        input_182 = None
        input_184 = torch.conv2d(
            input_183,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_183 = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_185 = torch.nn.functional.batch_norm(
            input_184,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_184 = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_186 = torch.nn.functional.silu(input_185, inplace=True)
        input_185 = None
        scale_105 = torch.nn.functional.adaptive_avg_pool2d(input_186, 1)
        scale_106 = torch.conv2d(
            scale_105,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_105 = l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_107 = torch.nn.functional.silu(scale_106, inplace=True)
        scale_106 = None
        scale_108 = torch.conv2d(
            scale_107,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_107 = l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_109 = torch.sigmoid(scale_108)
        scale_108 = None
        input_187 = scale_109 * input_186
        scale_109 = input_186 = None
        input_188 = torch.conv2d(
            input_187,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_187 = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_189 = torch.nn.functional.batch_norm(
            input_188,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_188 = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_17 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_17 = None
        input_189 += result_16
        result_17 = input_189
        input_189 = result_16 = None
        input_190 = torch.conv2d(
            result_17,
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
        input_191 = torch.nn.functional.batch_norm(
            input_190,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_190 = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_192 = torch.nn.functional.silu(input_191, inplace=True)
        input_191 = None
        input_193 = torch.conv2d(
            input_192,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_192 = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_194 = torch.nn.functional.batch_norm(
            input_193,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_193 = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_195 = torch.nn.functional.silu(input_194, inplace=True)
        input_194 = None
        scale_110 = torch.nn.functional.adaptive_avg_pool2d(input_195, 1)
        scale_111 = torch.conv2d(
            scale_110,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_110 = l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_112 = torch.nn.functional.silu(scale_111, inplace=True)
        scale_111 = None
        scale_113 = torch.conv2d(
            scale_112,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_112 = l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_114 = torch.sigmoid(scale_113)
        scale_113 = None
        input_196 = scale_114 * input_195
        scale_114 = input_195 = None
        input_197 = torch.conv2d(
            input_196,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_196 = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_198 = torch.nn.functional.batch_norm(
            input_197,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_197 = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_18 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_18 = None
        input_198 += result_17
        result_18 = input_198
        input_198 = result_17 = None
        input_199 = torch.conv2d(
            result_18,
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
        input_200 = torch.nn.functional.batch_norm(
            input_199,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_199 = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_201 = torch.nn.functional.silu(input_200, inplace=True)
        input_200 = None
        input_202 = torch.conv2d(
            input_201,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_201 = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_203 = torch.nn.functional.batch_norm(
            input_202,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_202 = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_204 = torch.nn.functional.silu(input_203, inplace=True)
        input_203 = None
        scale_115 = torch.nn.functional.adaptive_avg_pool2d(input_204, 1)
        scale_116 = torch.conv2d(
            scale_115,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_115 = l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_117 = torch.nn.functional.silu(scale_116, inplace=True)
        scale_116 = None
        scale_118 = torch.conv2d(
            scale_117,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_117 = l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_119 = torch.sigmoid(scale_118)
        scale_118 = None
        input_205 = scale_119 * input_204
        scale_119 = input_204 = None
        input_206 = torch.conv2d(
            input_205,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_205 = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_207 = torch.nn.functional.batch_norm(
            input_206,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_206 = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_19 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_19 = None
        input_207 += result_18
        result_19 = input_207
        input_207 = result_18 = None
        input_208 = torch.conv2d(
            result_19,
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
        input_209 = torch.nn.functional.batch_norm(
            input_208,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_208 = l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_210 = torch.nn.functional.silu(input_209, inplace=True)
        input_209 = None
        input_211 = torch.conv2d(
            input_210,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_210 = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_212 = torch.nn.functional.batch_norm(
            input_211,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_211 = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_213 = torch.nn.functional.silu(input_212, inplace=True)
        input_212 = None
        scale_120 = torch.nn.functional.adaptive_avg_pool2d(input_213, 1)
        scale_121 = torch.conv2d(
            scale_120,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_120 = l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_122 = torch.nn.functional.silu(scale_121, inplace=True)
        scale_121 = None
        scale_123 = torch.conv2d(
            scale_122,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_122 = l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_124 = torch.sigmoid(scale_123)
        scale_123 = None
        input_214 = scale_124 * input_213
        scale_124 = input_213 = None
        input_215 = torch.conv2d(
            input_214,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_214 = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_216 = torch.nn.functional.batch_norm(
            input_215,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_215 = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_20 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_20 = None
        input_216 += result_19
        result_20 = input_216
        input_216 = result_19 = None
        input_217 = torch.conv2d(
            result_20,
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
        input_218 = torch.nn.functional.batch_norm(
            input_217,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_217 = l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_219 = torch.nn.functional.silu(input_218, inplace=True)
        input_218 = None
        input_220 = torch.conv2d(
            input_219,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_219 = l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_221 = torch.nn.functional.batch_norm(
            input_220,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_220 = l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_222 = torch.nn.functional.silu(input_221, inplace=True)
        input_221 = None
        scale_125 = torch.nn.functional.adaptive_avg_pool2d(input_222, 1)
        scale_126 = torch.conv2d(
            scale_125,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_125 = l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_127 = torch.nn.functional.silu(scale_126, inplace=True)
        scale_126 = None
        scale_128 = torch.conv2d(
            scale_127,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_127 = l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_129 = torch.sigmoid(scale_128)
        scale_128 = None
        input_223 = scale_129 * input_222
        scale_129 = input_222 = None
        input_224 = torch.conv2d(
            input_223,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_223 = l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_225 = torch.nn.functional.batch_norm(
            input_224,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_224 = l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_7_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_21 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_21 = None
        input_225 += result_20
        result_21 = input_225
        input_225 = result_20 = None
        input_226 = torch.conv2d(
            result_21,
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
        input_227 = torch.nn.functional.batch_norm(
            input_226,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_226 = l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_228 = torch.nn.functional.silu(input_227, inplace=True)
        input_227 = None
        input_229 = torch.conv2d(
            input_228,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_228 = l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_230 = torch.nn.functional.batch_norm(
            input_229,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_229 = l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_231 = torch.nn.functional.silu(input_230, inplace=True)
        input_230 = None
        scale_130 = torch.nn.functional.adaptive_avg_pool2d(input_231, 1)
        scale_131 = torch.conv2d(
            scale_130,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_130 = l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_132 = torch.nn.functional.silu(scale_131, inplace=True)
        scale_131 = None
        scale_133 = torch.conv2d(
            scale_132,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_132 = l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_134 = torch.sigmoid(scale_133)
        scale_133 = None
        input_232 = scale_134 * input_231
        scale_134 = input_231 = None
        input_233 = torch.conv2d(
            input_232,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_232 = l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_234 = torch.nn.functional.batch_norm(
            input_233,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_233 = l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_8_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_22 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_22 = None
        input_234 += result_21
        result_22 = input_234
        input_234 = result_21 = None
        input_235 = torch.conv2d(
            result_22,
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
        input_236 = torch.nn.functional.batch_norm(
            input_235,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_235 = l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_237 = torch.nn.functional.silu(input_236, inplace=True)
        input_236 = None
        input_238 = torch.conv2d(
            input_237,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            960,
        )
        input_237 = l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_239 = torch.nn.functional.batch_norm(
            input_238,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_238 = l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_240 = torch.nn.functional.silu(input_239, inplace=True)
        input_239 = None
        scale_135 = torch.nn.functional.adaptive_avg_pool2d(input_240, 1)
        scale_136 = torch.conv2d(
            scale_135,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_135 = l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_137 = torch.nn.functional.silu(scale_136, inplace=True)
        scale_136 = None
        scale_138 = torch.conv2d(
            scale_137,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_137 = l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_139 = torch.sigmoid(scale_138)
        scale_138 = None
        input_241 = scale_139 * input_240
        scale_139 = input_240 = None
        input_242 = torch.conv2d(
            input_241,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_241 = l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_243 = torch.nn.functional.batch_norm(
            input_242,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_242 = l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_9_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_23 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_23 = None
        input_243 += result_22
        result_23 = input_243
        input_243 = result_22 = None
        input_244 = torch.conv2d(
            result_23,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_23 = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_245 = torch.nn.functional.batch_norm(
            input_244,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_244 = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_246 = torch.nn.functional.silu(input_245, inplace=True)
        input_245 = None
        input_247 = torch.conv2d(
            input_246,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            960,
        )
        input_246 = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_248 = torch.nn.functional.batch_norm(
            input_247,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_247 = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_249 = torch.nn.functional.silu(input_248, inplace=True)
        input_248 = None
        scale_140 = torch.nn.functional.adaptive_avg_pool2d(input_249, 1)
        scale_141 = torch.conv2d(
            scale_140,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_140 = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_142 = torch.nn.functional.silu(scale_141, inplace=True)
        scale_141 = None
        scale_143 = torch.conv2d(
            scale_142,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_142 = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_144 = torch.sigmoid(scale_143)
        scale_143 = None
        input_250 = scale_144 * input_249
        scale_144 = input_249 = None
        input_251 = torch.conv2d(
            input_250,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_250 = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_252 = torch.nn.functional.batch_norm(
            input_251,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_251 = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_253 = torch.conv2d(
            input_252,
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
        input_254 = torch.nn.functional.batch_norm(
            input_253,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_253 = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_255 = torch.nn.functional.silu(input_254, inplace=True)
        input_254 = None
        input_256 = torch.conv2d(
            input_255,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1344,
        )
        input_255 = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_257 = torch.nn.functional.batch_norm(
            input_256,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_256 = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_258 = torch.nn.functional.silu(input_257, inplace=True)
        input_257 = None
        scale_145 = torch.nn.functional.adaptive_avg_pool2d(input_258, 1)
        scale_146 = torch.conv2d(
            scale_145,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_145 = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_147 = torch.nn.functional.silu(scale_146, inplace=True)
        scale_146 = None
        scale_148 = torch.conv2d(
            scale_147,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_147 = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_149 = torch.sigmoid(scale_148)
        scale_148 = None
        input_259 = scale_149 * input_258
        scale_149 = input_258 = None
        input_260 = torch.conv2d(
            input_259,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_259 = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_261 = torch.nn.functional.batch_norm(
            input_260,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_260 = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_24 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_24 = None
        input_261 += input_252
        result_24 = input_261
        input_261 = input_252 = None
        input_262 = torch.conv2d(
            result_24,
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
        input_263 = torch.nn.functional.batch_norm(
            input_262,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_262 = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_264 = torch.nn.functional.silu(input_263, inplace=True)
        input_263 = None
        input_265 = torch.conv2d(
            input_264,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1344,
        )
        input_264 = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_266 = torch.nn.functional.batch_norm(
            input_265,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_265 = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_267 = torch.nn.functional.silu(input_266, inplace=True)
        input_266 = None
        scale_150 = torch.nn.functional.adaptive_avg_pool2d(input_267, 1)
        scale_151 = torch.conv2d(
            scale_150,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_150 = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_152 = torch.nn.functional.silu(scale_151, inplace=True)
        scale_151 = None
        scale_153 = torch.conv2d(
            scale_152,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_152 = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_154 = torch.sigmoid(scale_153)
        scale_153 = None
        input_268 = scale_154 * input_267
        scale_154 = input_267 = None
        input_269 = torch.conv2d(
            input_268,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_268 = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_270 = torch.nn.functional.batch_norm(
            input_269,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_269 = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_25 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_25 = None
        input_270 += result_24
        result_25 = input_270
        input_270 = result_24 = None
        input_271 = torch.conv2d(
            result_25,
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
        input_272 = torch.nn.functional.batch_norm(
            input_271,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_271 = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_273 = torch.nn.functional.silu(input_272, inplace=True)
        input_272 = None
        input_274 = torch.conv2d(
            input_273,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1344,
        )
        input_273 = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_275 = torch.nn.functional.batch_norm(
            input_274,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_274 = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_276 = torch.nn.functional.silu(input_275, inplace=True)
        input_275 = None
        scale_155 = torch.nn.functional.adaptive_avg_pool2d(input_276, 1)
        scale_156 = torch.conv2d(
            scale_155,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_155 = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_157 = torch.nn.functional.silu(scale_156, inplace=True)
        scale_156 = None
        scale_158 = torch.conv2d(
            scale_157,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_157 = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_159 = torch.sigmoid(scale_158)
        scale_158 = None
        input_277 = scale_159 * input_276
        scale_159 = input_276 = None
        input_278 = torch.conv2d(
            input_277,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_277 = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_279 = torch.nn.functional.batch_norm(
            input_278,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_278 = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_26 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_26 = None
        input_279 += result_25
        result_26 = input_279
        input_279 = result_25 = None
        input_280 = torch.conv2d(
            result_26,
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
        input_281 = torch.nn.functional.batch_norm(
            input_280,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_280 = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_282 = torch.nn.functional.silu(input_281, inplace=True)
        input_281 = None
        input_283 = torch.conv2d(
            input_282,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1344,
        )
        input_282 = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_284 = torch.nn.functional.batch_norm(
            input_283,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_283 = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_285 = torch.nn.functional.silu(input_284, inplace=True)
        input_284 = None
        scale_160 = torch.nn.functional.adaptive_avg_pool2d(input_285, 1)
        scale_161 = torch.conv2d(
            scale_160,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_160 = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_162 = torch.nn.functional.silu(scale_161, inplace=True)
        scale_161 = None
        scale_163 = torch.conv2d(
            scale_162,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_162 = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_164 = torch.sigmoid(scale_163)
        scale_163 = None
        input_286 = scale_164 * input_285
        scale_164 = input_285 = None
        input_287 = torch.conv2d(
            input_286,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_286 = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_288 = torch.nn.functional.batch_norm(
            input_287,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_287 = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_27 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_27 = None
        input_288 += result_26
        result_27 = input_288
        input_288 = result_26 = None
        input_289 = torch.conv2d(
            result_27,
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
        input_290 = torch.nn.functional.batch_norm(
            input_289,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_289 = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_291 = torch.nn.functional.silu(input_290, inplace=True)
        input_290 = None
        input_292 = torch.conv2d(
            input_291,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1344,
        )
        input_291 = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_293 = torch.nn.functional.batch_norm(
            input_292,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_292 = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_294 = torch.nn.functional.silu(input_293, inplace=True)
        input_293 = None
        scale_165 = torch.nn.functional.adaptive_avg_pool2d(input_294, 1)
        scale_166 = torch.conv2d(
            scale_165,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_165 = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_167 = torch.nn.functional.silu(scale_166, inplace=True)
        scale_166 = None
        scale_168 = torch.conv2d(
            scale_167,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_167 = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_169 = torch.sigmoid(scale_168)
        scale_168 = None
        input_295 = scale_169 * input_294
        scale_169 = input_294 = None
        input_296 = torch.conv2d(
            input_295,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_295 = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_297 = torch.nn.functional.batch_norm(
            input_296,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_296 = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_28 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_28 = None
        input_297 += result_27
        result_28 = input_297
        input_297 = result_27 = None
        input_298 = torch.conv2d(
            result_28,
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
        input_299 = torch.nn.functional.batch_norm(
            input_298,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_298 = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_300 = torch.nn.functional.silu(input_299, inplace=True)
        input_299 = None
        input_301 = torch.conv2d(
            input_300,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1344,
        )
        input_300 = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_302 = torch.nn.functional.batch_norm(
            input_301,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_301 = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_303 = torch.nn.functional.silu(input_302, inplace=True)
        input_302 = None
        scale_170 = torch.nn.functional.adaptive_avg_pool2d(input_303, 1)
        scale_171 = torch.conv2d(
            scale_170,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_170 = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_172 = torch.nn.functional.silu(scale_171, inplace=True)
        scale_171 = None
        scale_173 = torch.conv2d(
            scale_172,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_172 = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_174 = torch.sigmoid(scale_173)
        scale_173 = None
        input_304 = scale_174 * input_303
        scale_174 = input_303 = None
        input_305 = torch.conv2d(
            input_304,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_304 = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_306 = torch.nn.functional.batch_norm(
            input_305,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_305 = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_29 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_29 = None
        input_306 += result_28
        result_29 = input_306
        input_306 = result_28 = None
        input_307 = torch.conv2d(
            result_29,
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
        input_308 = torch.nn.functional.batch_norm(
            input_307,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_307 = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_309 = torch.nn.functional.silu(input_308, inplace=True)
        input_308 = None
        input_310 = torch.conv2d(
            input_309,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1344,
        )
        input_309 = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_311 = torch.nn.functional.batch_norm(
            input_310,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_310 = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_312 = torch.nn.functional.silu(input_311, inplace=True)
        input_311 = None
        scale_175 = torch.nn.functional.adaptive_avg_pool2d(input_312, 1)
        scale_176 = torch.conv2d(
            scale_175,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_175 = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_177 = torch.nn.functional.silu(scale_176, inplace=True)
        scale_176 = None
        scale_178 = torch.conv2d(
            scale_177,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_177 = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_179 = torch.sigmoid(scale_178)
        scale_178 = None
        input_313 = scale_179 * input_312
        scale_179 = input_312 = None
        input_314 = torch.conv2d(
            input_313,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_313 = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_315 = torch.nn.functional.batch_norm(
            input_314,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_314 = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_7_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_30 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_30 = None
        input_315 += result_29
        result_30 = input_315
        input_315 = result_29 = None
        input_316 = torch.conv2d(
            result_30,
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
        input_317 = torch.nn.functional.batch_norm(
            input_316,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_316 = l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_318 = torch.nn.functional.silu(input_317, inplace=True)
        input_317 = None
        input_319 = torch.conv2d(
            input_318,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1344,
        )
        input_318 = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_320 = torch.nn.functional.batch_norm(
            input_319,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_319 = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_321 = torch.nn.functional.silu(input_320, inplace=True)
        input_320 = None
        scale_180 = torch.nn.functional.adaptive_avg_pool2d(input_321, 1)
        scale_181 = torch.conv2d(
            scale_180,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_180 = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_182 = torch.nn.functional.silu(scale_181, inplace=True)
        scale_181 = None
        scale_183 = torch.conv2d(
            scale_182,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_182 = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_184 = torch.sigmoid(scale_183)
        scale_183 = None
        input_322 = scale_184 * input_321
        scale_184 = input_321 = None
        input_323 = torch.conv2d(
            input_322,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_322 = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_324 = torch.nn.functional.batch_norm(
            input_323,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_323 = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_8_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_31 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_31 = None
        input_324 += result_30
        result_31 = input_324
        input_324 = result_30 = None
        input_325 = torch.conv2d(
            result_31,
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
        input_326 = torch.nn.functional.batch_norm(
            input_325,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_325 = l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_327 = torch.nn.functional.silu(input_326, inplace=True)
        input_326 = None
        input_328 = torch.conv2d(
            input_327,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1344,
        )
        input_327 = l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_329 = torch.nn.functional.batch_norm(
            input_328,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_328 = l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_330 = torch.nn.functional.silu(input_329, inplace=True)
        input_329 = None
        scale_185 = torch.nn.functional.adaptive_avg_pool2d(input_330, 1)
        scale_186 = torch.conv2d(
            scale_185,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_185 = l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_187 = torch.nn.functional.silu(scale_186, inplace=True)
        scale_186 = None
        scale_188 = torch.conv2d(
            scale_187,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_187 = l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_189 = torch.sigmoid(scale_188)
        scale_188 = None
        input_331 = scale_189 * input_330
        scale_189 = input_330 = None
        input_332 = torch.conv2d(
            input_331,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_331 = l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_333 = torch.nn.functional.batch_norm(
            input_332,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_332 = l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_9_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_32 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_32 = None
        input_333 += result_31
        result_32 = input_333
        input_333 = result_31 = None
        input_334 = torch.conv2d(
            result_32,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_32 = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_335 = torch.nn.functional.batch_norm(
            input_334,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_334 = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_336 = torch.nn.functional.silu(input_335, inplace=True)
        input_335 = None
        input_337 = torch.conv2d(
            input_336,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            1344,
        )
        input_336 = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_338 = torch.nn.functional.batch_norm(
            input_337,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_337 = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_339 = torch.nn.functional.silu(input_338, inplace=True)
        input_338 = None
        scale_190 = torch.nn.functional.adaptive_avg_pool2d(input_339, 1)
        scale_191 = torch.conv2d(
            scale_190,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_190 = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_192 = torch.nn.functional.silu(scale_191, inplace=True)
        scale_191 = None
        scale_193 = torch.conv2d(
            scale_192,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_192 = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_194 = torch.sigmoid(scale_193)
        scale_193 = None
        input_340 = scale_194 * input_339
        scale_194 = input_339 = None
        input_341 = torch.conv2d(
            input_340,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_340 = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_342 = torch.nn.functional.batch_norm(
            input_341,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_341 = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_343 = torch.conv2d(
            input_342,
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
        input_344 = torch.nn.functional.batch_norm(
            input_343,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_343 = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_345 = torch.nn.functional.silu(input_344, inplace=True)
        input_344 = None
        input_346 = torch.conv2d(
            input_345,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2304,
        )
        input_345 = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_347 = torch.nn.functional.batch_norm(
            input_346,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_346 = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_348 = torch.nn.functional.silu(input_347, inplace=True)
        input_347 = None
        scale_195 = torch.nn.functional.adaptive_avg_pool2d(input_348, 1)
        scale_196 = torch.conv2d(
            scale_195,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_195 = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_197 = torch.nn.functional.silu(scale_196, inplace=True)
        scale_196 = None
        scale_198 = torch.conv2d(
            scale_197,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_197 = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_199 = torch.sigmoid(scale_198)
        scale_198 = None
        input_349 = scale_199 * input_348
        scale_199 = input_348 = None
        input_350 = torch.conv2d(
            input_349,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_349 = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_351 = torch.nn.functional.batch_norm(
            input_350,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_350 = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_33 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_33 = None
        input_351 += input_342
        result_33 = input_351
        input_351 = input_342 = None
        input_352 = torch.conv2d(
            result_33,
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
        input_353 = torch.nn.functional.batch_norm(
            input_352,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_352 = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_354 = torch.nn.functional.silu(input_353, inplace=True)
        input_353 = None
        input_355 = torch.conv2d(
            input_354,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2304,
        )
        input_354 = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_356 = torch.nn.functional.batch_norm(
            input_355,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_355 = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_357 = torch.nn.functional.silu(input_356, inplace=True)
        input_356 = None
        scale_200 = torch.nn.functional.adaptive_avg_pool2d(input_357, 1)
        scale_201 = torch.conv2d(
            scale_200,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_200 = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_202 = torch.nn.functional.silu(scale_201, inplace=True)
        scale_201 = None
        scale_203 = torch.conv2d(
            scale_202,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_202 = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_204 = torch.sigmoid(scale_203)
        scale_203 = None
        input_358 = scale_204 * input_357
        scale_204 = input_357 = None
        input_359 = torch.conv2d(
            input_358,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_358 = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_360 = torch.nn.functional.batch_norm(
            input_359,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_359 = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_34 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_34 = None
        input_360 += result_33
        result_34 = input_360
        input_360 = result_33 = None
        input_361 = torch.conv2d(
            result_34,
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
        input_362 = torch.nn.functional.batch_norm(
            input_361,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_361 = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_363 = torch.nn.functional.silu(input_362, inplace=True)
        input_362 = None
        input_364 = torch.conv2d(
            input_363,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2304,
        )
        input_363 = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_365 = torch.nn.functional.batch_norm(
            input_364,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_364 = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_366 = torch.nn.functional.silu(input_365, inplace=True)
        input_365 = None
        scale_205 = torch.nn.functional.adaptive_avg_pool2d(input_366, 1)
        scale_206 = torch.conv2d(
            scale_205,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_205 = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_207 = torch.nn.functional.silu(scale_206, inplace=True)
        scale_206 = None
        scale_208 = torch.conv2d(
            scale_207,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_207 = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_209 = torch.sigmoid(scale_208)
        scale_208 = None
        input_367 = scale_209 * input_366
        scale_209 = input_366 = None
        input_368 = torch.conv2d(
            input_367,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_367 = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_369 = torch.nn.functional.batch_norm(
            input_368,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_368 = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_35 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_35 = None
        input_369 += result_34
        result_35 = input_369
        input_369 = result_34 = None
        input_370 = torch.conv2d(
            result_35,
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
        input_371 = torch.nn.functional.batch_norm(
            input_370,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_370 = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_372 = torch.nn.functional.silu(input_371, inplace=True)
        input_371 = None
        input_373 = torch.conv2d(
            input_372,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2304,
        )
        input_372 = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_374 = torch.nn.functional.batch_norm(
            input_373,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_373 = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_375 = torch.nn.functional.silu(input_374, inplace=True)
        input_374 = None
        scale_210 = torch.nn.functional.adaptive_avg_pool2d(input_375, 1)
        scale_211 = torch.conv2d(
            scale_210,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_210 = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_212 = torch.nn.functional.silu(scale_211, inplace=True)
        scale_211 = None
        scale_213 = torch.conv2d(
            scale_212,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_212 = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_214 = torch.sigmoid(scale_213)
        scale_213 = None
        input_376 = scale_214 * input_375
        scale_214 = input_375 = None
        input_377 = torch.conv2d(
            input_376,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_376 = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_378 = torch.nn.functional.batch_norm(
            input_377,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_377 = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_36 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_36 = None
        input_378 += result_35
        result_36 = input_378
        input_378 = result_35 = None
        input_379 = torch.conv2d(
            result_36,
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
        input_380 = torch.nn.functional.batch_norm(
            input_379,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_379 = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_381 = torch.nn.functional.silu(input_380, inplace=True)
        input_380 = None
        input_382 = torch.conv2d(
            input_381,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2304,
        )
        input_381 = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_383 = torch.nn.functional.batch_norm(
            input_382,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_382 = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_384 = torch.nn.functional.silu(input_383, inplace=True)
        input_383 = None
        scale_215 = torch.nn.functional.adaptive_avg_pool2d(input_384, 1)
        scale_216 = torch.conv2d(
            scale_215,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_215 = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_217 = torch.nn.functional.silu(scale_216, inplace=True)
        scale_216 = None
        scale_218 = torch.conv2d(
            scale_217,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_217 = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_219 = torch.sigmoid(scale_218)
        scale_218 = None
        input_385 = scale_219 * input_384
        scale_219 = input_384 = None
        input_386 = torch.conv2d(
            input_385,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_385 = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_387 = torch.nn.functional.batch_norm(
            input_386,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_386 = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_37 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_37 = None
        input_387 += result_36
        result_37 = input_387
        input_387 = result_36 = None
        input_388 = torch.conv2d(
            result_37,
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
        input_389 = torch.nn.functional.batch_norm(
            input_388,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_388 = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_390 = torch.nn.functional.silu(input_389, inplace=True)
        input_389 = None
        input_391 = torch.conv2d(
            input_390,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2304,
        )
        input_390 = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_392 = torch.nn.functional.batch_norm(
            input_391,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_391 = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_393 = torch.nn.functional.silu(input_392, inplace=True)
        input_392 = None
        scale_220 = torch.nn.functional.adaptive_avg_pool2d(input_393, 1)
        scale_221 = torch.conv2d(
            scale_220,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_220 = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_222 = torch.nn.functional.silu(scale_221, inplace=True)
        scale_221 = None
        scale_223 = torch.conv2d(
            scale_222,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_222 = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_224 = torch.sigmoid(scale_223)
        scale_223 = None
        input_394 = scale_224 * input_393
        scale_224 = input_393 = None
        input_395 = torch.conv2d(
            input_394,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_394 = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_396 = torch.nn.functional.batch_norm(
            input_395,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_395 = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_38 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_38 = None
        input_396 += result_37
        result_38 = input_396
        input_396 = result_37 = None
        input_397 = torch.conv2d(
            result_38,
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
        input_398 = torch.nn.functional.batch_norm(
            input_397,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_397 = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_399 = torch.nn.functional.silu(input_398, inplace=True)
        input_398 = None
        input_400 = torch.conv2d(
            input_399,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2304,
        )
        input_399 = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_401 = torch.nn.functional.batch_norm(
            input_400,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_400 = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_402 = torch.nn.functional.silu(input_401, inplace=True)
        input_401 = None
        scale_225 = torch.nn.functional.adaptive_avg_pool2d(input_402, 1)
        scale_226 = torch.conv2d(
            scale_225,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_225 = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_227 = torch.nn.functional.silu(scale_226, inplace=True)
        scale_226 = None
        scale_228 = torch.conv2d(
            scale_227,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_227 = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_229 = torch.sigmoid(scale_228)
        scale_228 = None
        input_403 = scale_229 * input_402
        scale_229 = input_402 = None
        input_404 = torch.conv2d(
            input_403,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_403 = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_405 = torch.nn.functional.batch_norm(
            input_404,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_404 = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_7_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_39 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_39 = None
        input_405 += result_38
        result_39 = input_405
        input_405 = result_38 = None
        input_406 = torch.conv2d(
            result_39,
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
        input_407 = torch.nn.functional.batch_norm(
            input_406,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_406 = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_408 = torch.nn.functional.silu(input_407, inplace=True)
        input_407 = None
        input_409 = torch.conv2d(
            input_408,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2304,
        )
        input_408 = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_410 = torch.nn.functional.batch_norm(
            input_409,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_409 = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_411 = torch.nn.functional.silu(input_410, inplace=True)
        input_410 = None
        scale_230 = torch.nn.functional.adaptive_avg_pool2d(input_411, 1)
        scale_231 = torch.conv2d(
            scale_230,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_230 = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_232 = torch.nn.functional.silu(scale_231, inplace=True)
        scale_231 = None
        scale_233 = torch.conv2d(
            scale_232,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_232 = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_234 = torch.sigmoid(scale_233)
        scale_233 = None
        input_412 = scale_234 * input_411
        scale_234 = input_411 = None
        input_413 = torch.conv2d(
            input_412,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_412 = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_414 = torch.nn.functional.batch_norm(
            input_413,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_413 = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_8_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_40 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_40 = None
        input_414 += result_39
        result_40 = input_414
        input_414 = result_39 = None
        input_415 = torch.conv2d(
            result_40,
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
        input_416 = torch.nn.functional.batch_norm(
            input_415,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_415 = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_417 = torch.nn.functional.silu(input_416, inplace=True)
        input_416 = None
        input_418 = torch.conv2d(
            input_417,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2304,
        )
        input_417 = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_419 = torch.nn.functional.batch_norm(
            input_418,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_418 = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_420 = torch.nn.functional.silu(input_419, inplace=True)
        input_419 = None
        scale_235 = torch.nn.functional.adaptive_avg_pool2d(input_420, 1)
        scale_236 = torch.conv2d(
            scale_235,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_235 = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_237 = torch.nn.functional.silu(scale_236, inplace=True)
        scale_236 = None
        scale_238 = torch.conv2d(
            scale_237,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_237 = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_239 = torch.sigmoid(scale_238)
        scale_238 = None
        input_421 = scale_239 * input_420
        scale_239 = input_420 = None
        input_422 = torch.conv2d(
            input_421,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_421 = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_423 = torch.nn.functional.batch_norm(
            input_422,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_422 = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_9_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_41 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_41 = None
        input_423 += result_40
        result_41 = input_423
        input_423 = result_40 = None
        input_424 = torch.conv2d(
            result_41,
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
        input_425 = torch.nn.functional.batch_norm(
            input_424,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_424 = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_426 = torch.nn.functional.silu(input_425, inplace=True)
        input_425 = None
        input_427 = torch.conv2d(
            input_426,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2304,
        )
        input_426 = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_428 = torch.nn.functional.batch_norm(
            input_427,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_427 = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_429 = torch.nn.functional.silu(input_428, inplace=True)
        input_428 = None
        scale_240 = torch.nn.functional.adaptive_avg_pool2d(input_429, 1)
        scale_241 = torch.conv2d(
            scale_240,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_240 = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_242 = torch.nn.functional.silu(scale_241, inplace=True)
        scale_241 = None
        scale_243 = torch.conv2d(
            scale_242,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_242 = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_244 = torch.sigmoid(scale_243)
        scale_243 = None
        input_430 = scale_244 * input_429
        scale_244 = input_429 = None
        input_431 = torch.conv2d(
            input_430,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_430 = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_432 = torch.nn.functional.batch_norm(
            input_431,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_431 = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_10_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_42 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_42 = None
        input_432 += result_41
        result_42 = input_432
        input_432 = result_41 = None
        input_433 = torch.conv2d(
            result_42,
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
        input_434 = torch.nn.functional.batch_norm(
            input_433,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_433 = l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_435 = torch.nn.functional.silu(input_434, inplace=True)
        input_434 = None
        input_436 = torch.conv2d(
            input_435,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2304,
        )
        input_435 = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_437 = torch.nn.functional.batch_norm(
            input_436,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_436 = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_438 = torch.nn.functional.silu(input_437, inplace=True)
        input_437 = None
        scale_245 = torch.nn.functional.adaptive_avg_pool2d(input_438, 1)
        scale_246 = torch.conv2d(
            scale_245,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_245 = l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_247 = torch.nn.functional.silu(scale_246, inplace=True)
        scale_246 = None
        scale_248 = torch.conv2d(
            scale_247,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_247 = l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_249 = torch.sigmoid(scale_248)
        scale_248 = None
        input_439 = scale_249 * input_438
        scale_249 = input_438 = None
        input_440 = torch.conv2d(
            input_439,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_439 = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_441 = torch.nn.functional.batch_norm(
            input_440,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_440 = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_11_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_43 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_43 = None
        input_441 += result_42
        result_43 = input_441
        input_441 = result_42 = None
        input_442 = torch.conv2d(
            result_43,
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
        input_443 = torch.nn.functional.batch_norm(
            input_442,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_442 = l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_444 = torch.nn.functional.silu(input_443, inplace=True)
        input_443 = None
        input_445 = torch.conv2d(
            input_444,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            2304,
        )
        input_444 = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_446 = torch.nn.functional.batch_norm(
            input_445,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_445 = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_447 = torch.nn.functional.silu(input_446, inplace=True)
        input_446 = None
        scale_250 = torch.nn.functional.adaptive_avg_pool2d(input_447, 1)
        scale_251 = torch.conv2d(
            scale_250,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_250 = l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_252 = torch.nn.functional.silu(scale_251, inplace=True)
        scale_251 = None
        scale_253 = torch.conv2d(
            scale_252,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_252 = l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_254 = torch.sigmoid(scale_253)
        scale_253 = None
        input_448 = scale_254 * input_447
        scale_254 = input_447 = None
        input_449 = torch.conv2d(
            input_448,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_448 = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_450 = torch.nn.functional.batch_norm(
            input_449,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_449 = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_12_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_44 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_44 = None
        input_450 += result_43
        result_44 = input_450
        input_450 = result_43 = None
        input_451 = torch.conv2d(
            result_44,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_44 = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_452 = torch.nn.functional.batch_norm(
            input_451,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_451 = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_453 = torch.nn.functional.silu(input_452, inplace=True)
        input_452 = None
        input_454 = torch.conv2d(
            input_453,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2304,
        )
        input_453 = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_455 = torch.nn.functional.batch_norm(
            input_454,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_454 = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_456 = torch.nn.functional.silu(input_455, inplace=True)
        input_455 = None
        scale_255 = torch.nn.functional.adaptive_avg_pool2d(input_456, 1)
        scale_256 = torch.conv2d(
            scale_255,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_255 = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_257 = torch.nn.functional.silu(scale_256, inplace=True)
        scale_256 = None
        scale_258 = torch.conv2d(
            scale_257,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_257 = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_259 = torch.sigmoid(scale_258)
        scale_258 = None
        input_457 = scale_259 * input_456
        scale_259 = input_456 = None
        input_458 = torch.conv2d(
            input_457,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_457 = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_459 = torch.nn.functional.batch_norm(
            input_458,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_458 = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_0_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_460 = torch.conv2d(
            input_459,
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
        input_461 = torch.nn.functional.batch_norm(
            input_460,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_460 = l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_462 = torch.nn.functional.silu(input_461, inplace=True)
        input_461 = None
        input_463 = torch.conv2d(
            input_462,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3840,
        )
        input_462 = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_464 = torch.nn.functional.batch_norm(
            input_463,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_463 = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_465 = torch.nn.functional.silu(input_464, inplace=True)
        input_464 = None
        scale_260 = torch.nn.functional.adaptive_avg_pool2d(input_465, 1)
        scale_261 = torch.conv2d(
            scale_260,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_260 = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_262 = torch.nn.functional.silu(scale_261, inplace=True)
        scale_261 = None
        scale_263 = torch.conv2d(
            scale_262,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_262 = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_264 = torch.sigmoid(scale_263)
        scale_263 = None
        input_466 = scale_264 * input_465
        scale_264 = input_465 = None
        input_467 = torch.conv2d(
            input_466,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_466 = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_468 = torch.nn.functional.batch_norm(
            input_467,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_467 = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_1_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_45 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_45 = None
        input_468 += input_459
        result_45 = input_468
        input_468 = input_459 = None
        input_469 = torch.conv2d(
            result_45,
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
        input_470 = torch.nn.functional.batch_norm(
            input_469,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_469 = l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_471 = torch.nn.functional.silu(input_470, inplace=True)
        input_470 = None
        input_472 = torch.conv2d(
            input_471,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3840,
        )
        input_471 = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_473 = torch.nn.functional.batch_norm(
            input_472,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_472 = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_474 = torch.nn.functional.silu(input_473, inplace=True)
        input_473 = None
        scale_265 = torch.nn.functional.adaptive_avg_pool2d(input_474, 1)
        scale_266 = torch.conv2d(
            scale_265,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_265 = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_267 = torch.nn.functional.silu(scale_266, inplace=True)
        scale_266 = None
        scale_268 = torch.conv2d(
            scale_267,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_267 = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_269 = torch.sigmoid(scale_268)
        scale_268 = None
        input_475 = scale_269 * input_474
        scale_269 = input_474 = None
        input_476 = torch.conv2d(
            input_475,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_475 = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_477 = torch.nn.functional.batch_norm(
            input_476,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_476 = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_2_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_46 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_46 = None
        input_477 += result_45
        result_46 = input_477
        input_477 = result_45 = None
        input_478 = torch.conv2d(
            result_46,
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
        input_479 = torch.nn.functional.batch_norm(
            input_478,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_478 = l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_480 = torch.nn.functional.silu(input_479, inplace=True)
        input_479 = None
        input_481 = torch.conv2d(
            input_480,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3840,
        )
        input_480 = l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_482 = torch.nn.functional.batch_norm(
            input_481,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_481 = l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_483 = torch.nn.functional.silu(input_482, inplace=True)
        input_482 = None
        scale_270 = torch.nn.functional.adaptive_avg_pool2d(input_483, 1)
        scale_271 = torch.conv2d(
            scale_270,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_270 = l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_272 = torch.nn.functional.silu(scale_271, inplace=True)
        scale_271 = None
        scale_273 = torch.conv2d(
            scale_272,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_272 = l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_274 = torch.sigmoid(scale_273)
        scale_273 = None
        input_484 = scale_274 * input_483
        scale_274 = input_483 = None
        input_485 = torch.conv2d(
            input_484,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_484 = l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_486 = torch.nn.functional.batch_norm(
            input_485,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_485 = l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_3_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        _log_api_usage_once_47 = torch._C._log_api_usage_once(
            "torchvision.ops.stochastic_depth.stochastic_depth"
        )
        _log_api_usage_once_47 = None
        input_486 += result_46
        result_47 = input_486
        input_486 = result_46 = None
        input_487 = torch.conv2d(
            result_47,
            l_self_modules_features_modules_8_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_47 = (
            l_self_modules_features_modules_8_modules_0_parameters_weight_
        ) = None
        input_488 = torch.nn.functional.batch_norm(
            input_487,
            l_self_modules_features_modules_8_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_1_buffers_running_var_,
            l_self_modules_features_modules_8_modules_1_parameters_weight_,
            l_self_modules_features_modules_8_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_487 = (
            l_self_modules_features_modules_8_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_8_modules_1_buffers_running_var_
        ) = (
            l_self_modules_features_modules_8_modules_1_parameters_weight_
        ) = l_self_modules_features_modules_8_modules_1_parameters_bias_ = None
        input_489 = torch.nn.functional.silu(input_488, inplace=True)
        input_488 = None
        x = torch.nn.functional.adaptive_avg_pool2d(input_489, 1)
        input_489 = None
        x_1 = torch.flatten(x, 1)
        x = None
        input_490 = torch.nn.functional.dropout(x_1, 0.5, False, True)
        x_1 = None
        input_491 = torch._C._nn.linear(
            input_490,
            l_self_modules_classifier_modules_1_parameters_weight_,
            l_self_modules_classifier_modules_1_parameters_bias_,
        )
        input_490 = (
            l_self_modules_classifier_modules_1_parameters_weight_
        ) = l_self_modules_classifier_modules_1_parameters_bias_ = None
        return (input_491,)
