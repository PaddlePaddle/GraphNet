import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_backbone_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_1_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_1_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_2_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_2_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_2_modules_block_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_3_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_3_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_3_modules_block_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_4_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_4_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_4_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_5_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_5_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_5_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_6_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_6_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_6_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_7_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_7_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_7_modules_block_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_8_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_8_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_8_modules_block_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_9_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_9_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_9_modules_block_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_10_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_10_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_10_modules_block_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_11_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_11_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_11_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_12_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_12_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_12_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_13_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_13_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_13_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_14_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_14_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_14_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_15_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_15_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_15_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_16_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_16_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_16_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_16_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_16_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_modules_convs_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_modules_convs_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_modules_convs_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_modules_convs_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_modules_convs_modules_4_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_modules_project_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_modules_project_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_classifier_modules_0_modules_project_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_classifier_modules_0_modules_project_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_modules_project_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_classifier_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_classifier_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_aux_classifier_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_aux_classifier_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_aux_classifier_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_aux_classifier_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_aux_classifier_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_aux_classifier_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_aux_classifier_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_backbone_modules_0_modules_0_parameters_weight_ = (
            L_self_modules_backbone_modules_0_modules_0_parameters_weight_
        )
        l_self_modules_backbone_modules_0_modules_1_buffers_running_mean_ = (
            L_self_modules_backbone_modules_0_modules_1_buffers_running_mean_
        )
        l_self_modules_backbone_modules_0_modules_1_buffers_running_var_ = (
            L_self_modules_backbone_modules_0_modules_1_buffers_running_var_
        )
        l_self_modules_backbone_modules_0_modules_1_parameters_weight_ = (
            L_self_modules_backbone_modules_0_modules_1_parameters_weight_
        )
        l_self_modules_backbone_modules_0_modules_1_parameters_bias_ = (
            L_self_modules_backbone_modules_0_modules_1_parameters_bias_
        )
        l_self_modules_backbone_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_1_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_backbone_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_1_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_2_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_backbone_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_2_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_2_modules_block_modules_2_modules_0_parameters_weight_ = L_self_modules_backbone_modules_2_modules_block_modules_2_modules_0_parameters_weight_
        l_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_parameters_bias_ = L_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_parameters_bias_
        l_self_modules_backbone_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_3_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_backbone_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_3_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_3_modules_block_modules_2_modules_0_parameters_weight_ = L_self_modules_backbone_modules_3_modules_block_modules_2_modules_0_parameters_weight_
        l_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_parameters_bias_ = L_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_parameters_bias_
        l_self_modules_backbone_modules_4_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_4_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_backbone_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_4_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_4_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_5_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_5_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_backbone_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_5_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_5_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_6_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_6_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_backbone_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_6_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_6_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_7_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_7_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_backbone_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_7_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_7_modules_block_modules_2_modules_0_parameters_weight_ = L_self_modules_backbone_modules_7_modules_block_modules_2_modules_0_parameters_weight_
        l_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_parameters_bias_ = L_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_parameters_bias_
        l_self_modules_backbone_modules_8_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_8_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_backbone_modules_8_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_8_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_8_modules_block_modules_2_modules_0_parameters_weight_ = L_self_modules_backbone_modules_8_modules_block_modules_2_modules_0_parameters_weight_
        l_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_parameters_bias_ = L_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_parameters_bias_
        l_self_modules_backbone_modules_9_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_9_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_backbone_modules_9_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_9_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_9_modules_block_modules_2_modules_0_parameters_weight_ = L_self_modules_backbone_modules_9_modules_block_modules_2_modules_0_parameters_weight_
        l_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_parameters_bias_ = L_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_parameters_bias_
        l_self_modules_backbone_modules_10_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_10_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_backbone_modules_10_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_10_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_10_modules_block_modules_2_modules_0_parameters_weight_ = L_self_modules_backbone_modules_10_modules_block_modules_2_modules_0_parameters_weight_
        l_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_parameters_weight_ = L_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_parameters_weight_
        l_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_parameters_bias_ = L_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_parameters_bias_
        l_self_modules_backbone_modules_11_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_11_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_backbone_modules_11_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_11_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_11_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_11_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_12_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_12_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_backbone_modules_12_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_12_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_12_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_12_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_13_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_13_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_backbone_modules_13_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_13_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_13_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_13_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_14_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_14_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_backbone_modules_14_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_14_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_14_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_14_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_15_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_backbone_modules_15_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_backbone_modules_15_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_backbone_modules_15_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_backbone_modules_15_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_backbone_modules_15_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_backbone_modules_16_modules_0_parameters_weight_ = (
            L_self_modules_backbone_modules_16_modules_0_parameters_weight_
        )
        l_self_modules_backbone_modules_16_modules_1_buffers_running_mean_ = (
            L_self_modules_backbone_modules_16_modules_1_buffers_running_mean_
        )
        l_self_modules_backbone_modules_16_modules_1_buffers_running_var_ = (
            L_self_modules_backbone_modules_16_modules_1_buffers_running_var_
        )
        l_self_modules_backbone_modules_16_modules_1_parameters_weight_ = (
            L_self_modules_backbone_modules_16_modules_1_parameters_weight_
        )
        l_self_modules_backbone_modules_16_modules_1_parameters_bias_ = (
            L_self_modules_backbone_modules_16_modules_1_parameters_bias_
        )
        l_self_modules_classifier_modules_0_modules_convs_modules_0_modules_0_parameters_weight_ = L_self_modules_classifier_modules_0_modules_convs_modules_0_modules_0_parameters_weight_
        l_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_buffers_running_mean_ = L_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_buffers_running_mean_
        l_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_buffers_running_var_ = L_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_buffers_running_var_
        l_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_parameters_weight_ = L_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_parameters_weight_
        l_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_parameters_bias_ = L_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_parameters_bias_
        l_self_modules_classifier_modules_0_modules_convs_modules_1_modules_0_parameters_weight_ = L_self_modules_classifier_modules_0_modules_convs_modules_1_modules_0_parameters_weight_
        l_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_buffers_running_mean_ = L_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_buffers_running_mean_
        l_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_buffers_running_var_ = L_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_buffers_running_var_
        l_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_parameters_weight_ = L_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_parameters_weight_
        l_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_parameters_bias_ = L_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_parameters_bias_
        l_self_modules_classifier_modules_0_modules_convs_modules_2_modules_0_parameters_weight_ = L_self_modules_classifier_modules_0_modules_convs_modules_2_modules_0_parameters_weight_
        l_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_buffers_running_mean_ = L_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_buffers_running_mean_
        l_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_buffers_running_var_ = L_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_buffers_running_var_
        l_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_parameters_weight_ = L_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_parameters_weight_
        l_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_parameters_bias_ = L_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_parameters_bias_
        l_self_modules_classifier_modules_0_modules_convs_modules_3_modules_0_parameters_weight_ = L_self_modules_classifier_modules_0_modules_convs_modules_3_modules_0_parameters_weight_
        l_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_buffers_running_mean_ = L_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_buffers_running_mean_
        l_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_buffers_running_var_ = L_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_buffers_running_var_
        l_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_parameters_weight_ = L_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_parameters_weight_
        l_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_parameters_bias_ = L_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_parameters_bias_
        l_self_modules_classifier_modules_0_modules_convs_modules_4_modules_1_parameters_weight_ = L_self_modules_classifier_modules_0_modules_convs_modules_4_modules_1_parameters_weight_
        l_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_buffers_running_mean_ = L_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_buffers_running_mean_
        l_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_buffers_running_var_ = L_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_buffers_running_var_
        l_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_parameters_weight_ = L_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_parameters_weight_
        l_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_parameters_bias_ = L_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_parameters_bias_
        l_self_modules_classifier_modules_0_modules_project_modules_0_parameters_weight_ = L_self_modules_classifier_modules_0_modules_project_modules_0_parameters_weight_
        l_self_modules_classifier_modules_0_modules_project_modules_1_buffers_running_mean_ = L_self_modules_classifier_modules_0_modules_project_modules_1_buffers_running_mean_
        l_self_modules_classifier_modules_0_modules_project_modules_1_buffers_running_var_ = L_self_modules_classifier_modules_0_modules_project_modules_1_buffers_running_var_
        l_self_modules_classifier_modules_0_modules_project_modules_1_parameters_weight_ = L_self_modules_classifier_modules_0_modules_project_modules_1_parameters_weight_
        l_self_modules_classifier_modules_0_modules_project_modules_1_parameters_bias_ = L_self_modules_classifier_modules_0_modules_project_modules_1_parameters_bias_
        l_self_modules_classifier_modules_1_parameters_weight_ = (
            L_self_modules_classifier_modules_1_parameters_weight_
        )
        l_self_modules_classifier_modules_2_buffers_running_mean_ = (
            L_self_modules_classifier_modules_2_buffers_running_mean_
        )
        l_self_modules_classifier_modules_2_buffers_running_var_ = (
            L_self_modules_classifier_modules_2_buffers_running_var_
        )
        l_self_modules_classifier_modules_2_parameters_weight_ = (
            L_self_modules_classifier_modules_2_parameters_weight_
        )
        l_self_modules_classifier_modules_2_parameters_bias_ = (
            L_self_modules_classifier_modules_2_parameters_bias_
        )
        l_self_modules_classifier_modules_4_parameters_weight_ = (
            L_self_modules_classifier_modules_4_parameters_weight_
        )
        l_self_modules_classifier_modules_4_parameters_bias_ = (
            L_self_modules_classifier_modules_4_parameters_bias_
        )
        l_self_modules_aux_classifier_modules_0_parameters_weight_ = (
            L_self_modules_aux_classifier_modules_0_parameters_weight_
        )
        l_self_modules_aux_classifier_modules_1_buffers_running_mean_ = (
            L_self_modules_aux_classifier_modules_1_buffers_running_mean_
        )
        l_self_modules_aux_classifier_modules_1_buffers_running_var_ = (
            L_self_modules_aux_classifier_modules_1_buffers_running_var_
        )
        l_self_modules_aux_classifier_modules_1_parameters_weight_ = (
            L_self_modules_aux_classifier_modules_1_parameters_weight_
        )
        l_self_modules_aux_classifier_modules_1_parameters_bias_ = (
            L_self_modules_aux_classifier_modules_1_parameters_bias_
        )
        l_self_modules_aux_classifier_modules_4_parameters_weight_ = (
            L_self_modules_aux_classifier_modules_4_parameters_weight_
        )
        l_self_modules_aux_classifier_modules_4_parameters_bias_ = (
            L_self_modules_aux_classifier_modules_4_parameters_bias_
        )
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_backbone_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_backbone_modules_0_modules_0_parameters_weight_ = None
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_backbone_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_0_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_0_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_1 = (
            l_self_modules_backbone_modules_0_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_0_modules_1_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_0_modules_1_parameters_weight_
        ) = l_self_modules_backbone_modules_0_modules_1_parameters_bias_ = None
        input_3 = torch.nn.functional.hardswish(input_2, True)
        input_2 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_backbone_modules_1_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            16,
        )
        l_self_modules_backbone_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_4 = l_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_backbone_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_6 = torch.nn.functional.relu(input_5, inplace=True)
        input_5 = None
        input_7 = torch.conv2d(
            input_6,
            l_self_modules_backbone_modules_1_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_6 = l_self_modules_backbone_modules_1_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_7 = l_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_1_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_8 += input_3
        result = input_8
        input_8 = input_3 = None
        input_9 = torch.conv2d(
            result,
            l_self_modules_backbone_modules_2_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result = l_self_modules_backbone_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_10 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_9 = l_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_backbone_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_11 = torch.nn.functional.relu(input_10, inplace=True)
        input_10 = None
        input_12 = torch.conv2d(
            input_11,
            l_self_modules_backbone_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            64,
        )
        input_11 = l_self_modules_backbone_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_13 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_12 = l_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_14 = torch.nn.functional.relu(input_13, inplace=True)
        input_13 = None
        input_15 = torch.conv2d(
            input_14,
            l_self_modules_backbone_modules_2_modules_block_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_14 = l_self_modules_backbone_modules_2_modules_block_modules_2_modules_0_parameters_weight_ = (None)
        input_16 = torch.nn.functional.batch_norm(
            input_15,
            l_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_15 = l_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_parameters_weight_ = l_self_modules_backbone_modules_2_modules_block_modules_2_modules_1_parameters_bias_ = (None)
        input_17 = torch.conv2d(
            input_16,
            l_self_modules_backbone_modules_3_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_18 = torch.nn.functional.batch_norm(
            input_17,
            l_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_17 = l_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_backbone_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_19 = torch.nn.functional.relu(input_18, inplace=True)
        input_18 = None
        input_20 = torch.conv2d(
            input_19,
            l_self_modules_backbone_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            72,
        )
        input_19 = l_self_modules_backbone_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_21 = torch.nn.functional.batch_norm(
            input_20,
            l_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_20 = l_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_22 = torch.nn.functional.relu(input_21, inplace=True)
        input_21 = None
        input_23 = torch.conv2d(
            input_22,
            l_self_modules_backbone_modules_3_modules_block_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_22 = l_self_modules_backbone_modules_3_modules_block_modules_2_modules_0_parameters_weight_ = (None)
        input_24 = torch.nn.functional.batch_norm(
            input_23,
            l_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_23 = l_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_parameters_weight_ = l_self_modules_backbone_modules_3_modules_block_modules_2_modules_1_parameters_bias_ = (None)
        input_24 += input_16
        result_1 = input_24
        input_24 = input_16 = None
        input_25 = torch.conv2d(
            result_1,
            l_self_modules_backbone_modules_4_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_1 = l_self_modules_backbone_modules_4_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_26 = torch.nn.functional.batch_norm(
            input_25,
            l_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_25 = l_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_backbone_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_27 = torch.nn.functional.relu(input_26, inplace=True)
        input_26 = None
        input_28 = torch.conv2d(
            input_27,
            l_self_modules_backbone_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            72,
        )
        input_27 = l_self_modules_backbone_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_29 = torch.nn.functional.batch_norm(
            input_28,
            l_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_28 = l_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_30 = torch.nn.functional.relu(input_29, inplace=True)
        input_29 = None
        scale = torch.nn.functional.adaptive_avg_pool2d(input_30, 1)
        scale_1 = torch.conv2d(
            scale,
            l_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale = l_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_2 = torch.nn.functional.relu(scale_1, inplace=False)
        scale_1 = None
        scale_3 = torch.conv2d(
            scale_2,
            l_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_2 = l_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_4 = torch.nn.functional.hardsigmoid(scale_3, False)
        scale_3 = None
        input_31 = scale_4 * input_30
        scale_4 = input_30 = None
        input_32 = torch.conv2d(
            input_31,
            l_self_modules_backbone_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_31 = l_self_modules_backbone_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_33 = torch.nn.functional.batch_norm(
            input_32,
            l_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_32 = l_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_34 = torch.conv2d(
            input_33,
            l_self_modules_backbone_modules_5_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_5_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_35 = torch.nn.functional.batch_norm(
            input_34,
            l_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_34 = l_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_backbone_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_36 = torch.nn.functional.relu(input_35, inplace=True)
        input_35 = None
        input_37 = torch.conv2d(
            input_36,
            l_self_modules_backbone_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        input_36 = l_self_modules_backbone_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_38 = torch.nn.functional.batch_norm(
            input_37,
            l_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_37 = l_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_39 = torch.nn.functional.relu(input_38, inplace=True)
        input_38 = None
        scale_5 = torch.nn.functional.adaptive_avg_pool2d(input_39, 1)
        scale_6 = torch.conv2d(
            scale_5,
            l_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_5 = l_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_7 = torch.nn.functional.relu(scale_6, inplace=False)
        scale_6 = None
        scale_8 = torch.conv2d(
            scale_7,
            l_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_7 = l_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_9 = torch.nn.functional.hardsigmoid(scale_8, False)
        scale_8 = None
        input_40 = scale_9 * input_39
        scale_9 = input_39 = None
        input_41 = torch.conv2d(
            input_40,
            l_self_modules_backbone_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_40 = l_self_modules_backbone_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_42 = torch.nn.functional.batch_norm(
            input_41,
            l_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_41 = l_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_42 += input_33
        result_2 = input_42
        input_42 = None
        input_43 = torch.conv2d(
            result_2,
            l_self_modules_backbone_modules_6_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_6_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_44 = torch.nn.functional.batch_norm(
            input_43,
            l_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_43 = l_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_backbone_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_45 = torch.nn.functional.relu(input_44, inplace=True)
        input_44 = None
        input_46 = torch.conv2d(
            input_45,
            l_self_modules_backbone_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        input_45 = l_self_modules_backbone_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_47 = torch.nn.functional.batch_norm(
            input_46,
            l_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_46 = l_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_48 = torch.nn.functional.relu(input_47, inplace=True)
        input_47 = None
        scale_10 = torch.nn.functional.adaptive_avg_pool2d(input_48, 1)
        scale_11 = torch.conv2d(
            scale_10,
            l_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_10 = l_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_12 = torch.nn.functional.relu(scale_11, inplace=False)
        scale_11 = None
        scale_13 = torch.conv2d(
            scale_12,
            l_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_12 = l_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_14 = torch.nn.functional.hardsigmoid(scale_13, False)
        scale_13 = None
        input_49 = scale_14 * input_48
        scale_14 = input_48 = None
        input_50 = torch.conv2d(
            input_49,
            l_self_modules_backbone_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_49 = l_self_modules_backbone_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_51 = torch.nn.functional.batch_norm(
            input_50,
            l_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_50 = l_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_51 += result_2
        result_3 = input_51
        input_51 = result_2 = None
        input_52 = torch.conv2d(
            result_3,
            l_self_modules_backbone_modules_7_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_3 = l_self_modules_backbone_modules_7_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_53 = torch.nn.functional.batch_norm(
            input_52,
            l_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_52 = l_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_backbone_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_54 = torch.nn.functional.hardswish(input_53, True)
        input_53 = None
        input_55 = torch.conv2d(
            input_54,
            l_self_modules_backbone_modules_7_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            240,
        )
        input_54 = l_self_modules_backbone_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_56 = torch.nn.functional.batch_norm(
            input_55,
            l_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_55 = l_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_57 = torch.nn.functional.hardswish(input_56, True)
        input_56 = None
        input_58 = torch.conv2d(
            input_57,
            l_self_modules_backbone_modules_7_modules_block_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_57 = l_self_modules_backbone_modules_7_modules_block_modules_2_modules_0_parameters_weight_ = (None)
        input_59 = torch.nn.functional.batch_norm(
            input_58,
            l_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_58 = l_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_parameters_weight_ = l_self_modules_backbone_modules_7_modules_block_modules_2_modules_1_parameters_bias_ = (None)
        input_60 = torch.conv2d(
            input_59,
            l_self_modules_backbone_modules_8_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_8_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_61 = torch.nn.functional.batch_norm(
            input_60,
            l_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_60 = l_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_backbone_modules_8_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_62 = torch.nn.functional.hardswish(input_61, True)
        input_61 = None
        input_63 = torch.conv2d(
            input_62,
            l_self_modules_backbone_modules_8_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            200,
        )
        input_62 = l_self_modules_backbone_modules_8_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_64 = torch.nn.functional.batch_norm(
            input_63,
            l_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_63 = l_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_8_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_65 = torch.nn.functional.hardswish(input_64, True)
        input_64 = None
        input_66 = torch.conv2d(
            input_65,
            l_self_modules_backbone_modules_8_modules_block_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_65 = l_self_modules_backbone_modules_8_modules_block_modules_2_modules_0_parameters_weight_ = (None)
        input_67 = torch.nn.functional.batch_norm(
            input_66,
            l_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_66 = l_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_parameters_weight_ = l_self_modules_backbone_modules_8_modules_block_modules_2_modules_1_parameters_bias_ = (None)
        input_67 += input_59
        result_4 = input_67
        input_67 = input_59 = None
        input_68 = torch.conv2d(
            result_4,
            l_self_modules_backbone_modules_9_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_9_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_69 = torch.nn.functional.batch_norm(
            input_68,
            l_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_68 = l_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_backbone_modules_9_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_70 = torch.nn.functional.hardswish(input_69, True)
        input_69 = None
        input_71 = torch.conv2d(
            input_70,
            l_self_modules_backbone_modules_9_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            184,
        )
        input_70 = l_self_modules_backbone_modules_9_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_72 = torch.nn.functional.batch_norm(
            input_71,
            l_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_71 = l_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_9_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_73 = torch.nn.functional.hardswish(input_72, True)
        input_72 = None
        input_74 = torch.conv2d(
            input_73,
            l_self_modules_backbone_modules_9_modules_block_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_73 = l_self_modules_backbone_modules_9_modules_block_modules_2_modules_0_parameters_weight_ = (None)
        input_75 = torch.nn.functional.batch_norm(
            input_74,
            l_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_74 = l_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_parameters_weight_ = l_self_modules_backbone_modules_9_modules_block_modules_2_modules_1_parameters_bias_ = (None)
        input_75 += result_4
        result_5 = input_75
        input_75 = result_4 = None
        input_76 = torch.conv2d(
            result_5,
            l_self_modules_backbone_modules_10_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_10_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_77 = torch.nn.functional.batch_norm(
            input_76,
            l_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_76 = l_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_backbone_modules_10_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_78 = torch.nn.functional.hardswish(input_77, True)
        input_77 = None
        input_79 = torch.conv2d(
            input_78,
            l_self_modules_backbone_modules_10_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            184,
        )
        input_78 = l_self_modules_backbone_modules_10_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_80 = torch.nn.functional.batch_norm(
            input_79,
            l_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_79 = l_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_10_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_81 = torch.nn.functional.hardswish(input_80, True)
        input_80 = None
        input_82 = torch.conv2d(
            input_81,
            l_self_modules_backbone_modules_10_modules_block_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_81 = l_self_modules_backbone_modules_10_modules_block_modules_2_modules_0_parameters_weight_ = (None)
        input_83 = torch.nn.functional.batch_norm(
            input_82,
            l_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_82 = l_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_parameters_weight_ = l_self_modules_backbone_modules_10_modules_block_modules_2_modules_1_parameters_bias_ = (None)
        input_83 += result_5
        result_6 = input_83
        input_83 = result_5 = None
        input_84 = torch.conv2d(
            result_6,
            l_self_modules_backbone_modules_11_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_6 = l_self_modules_backbone_modules_11_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_85 = torch.nn.functional.batch_norm(
            input_84,
            l_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_84 = l_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_backbone_modules_11_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_86 = torch.nn.functional.hardswish(input_85, True)
        input_85 = None
        input_87 = torch.conv2d(
            input_86,
            l_self_modules_backbone_modules_11_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            480,
        )
        input_86 = l_self_modules_backbone_modules_11_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_88 = torch.nn.functional.batch_norm(
            input_87,
            l_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_87 = l_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_11_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_89 = torch.nn.functional.hardswish(input_88, True)
        input_88 = None
        scale_15 = torch.nn.functional.adaptive_avg_pool2d(input_89, 1)
        scale_16 = torch.conv2d(
            scale_15,
            l_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_15 = l_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_17 = torch.nn.functional.relu(scale_16, inplace=False)
        scale_16 = None
        scale_18 = torch.conv2d(
            scale_17,
            l_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_17 = l_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_19 = torch.nn.functional.hardsigmoid(scale_18, False)
        scale_18 = None
        input_90 = scale_19 * input_89
        scale_19 = input_89 = None
        input_91 = torch.conv2d(
            input_90,
            l_self_modules_backbone_modules_11_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_90 = l_self_modules_backbone_modules_11_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_92 = torch.nn.functional.batch_norm(
            input_91,
            l_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_91 = l_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_11_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_93 = torch.conv2d(
            input_92,
            l_self_modules_backbone_modules_12_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_12_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_94 = torch.nn.functional.batch_norm(
            input_93,
            l_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_93 = l_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_backbone_modules_12_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_95 = torch.nn.functional.hardswish(input_94, True)
        input_94 = None
        input_96 = torch.conv2d(
            input_95,
            l_self_modules_backbone_modules_12_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            672,
        )
        input_95 = l_self_modules_backbone_modules_12_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_97 = torch.nn.functional.batch_norm(
            input_96,
            l_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_96 = l_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_12_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_98 = torch.nn.functional.hardswish(input_97, True)
        input_97 = None
        scale_20 = torch.nn.functional.adaptive_avg_pool2d(input_98, 1)
        scale_21 = torch.conv2d(
            scale_20,
            l_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_20 = l_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_22 = torch.nn.functional.relu(scale_21, inplace=False)
        scale_21 = None
        scale_23 = torch.conv2d(
            scale_22,
            l_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_22 = l_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_12_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_24 = torch.nn.functional.hardsigmoid(scale_23, False)
        scale_23 = None
        input_99 = scale_24 * input_98
        scale_24 = input_98 = None
        input_100 = torch.conv2d(
            input_99,
            l_self_modules_backbone_modules_12_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_99 = l_self_modules_backbone_modules_12_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_101 = torch.nn.functional.batch_norm(
            input_100,
            l_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_100 = l_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_12_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_101 += input_92
        result_7 = input_101
        input_101 = input_92 = None
        input_102 = torch.conv2d(
            result_7,
            l_self_modules_backbone_modules_13_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_7 = l_self_modules_backbone_modules_13_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_103 = torch.nn.functional.batch_norm(
            input_102,
            l_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_102 = l_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_backbone_modules_13_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_104 = torch.nn.functional.hardswish(input_103, True)
        input_103 = None
        input_105 = torch.conv2d(
            input_104,
            l_self_modules_backbone_modules_13_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (2, 2),
            672,
        )
        input_104 = l_self_modules_backbone_modules_13_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_106 = torch.nn.functional.batch_norm(
            input_105,
            l_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_105 = l_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_13_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_107 = torch.nn.functional.hardswish(input_106, True)
        input_106 = None
        scale_25 = torch.nn.functional.adaptive_avg_pool2d(input_107, 1)
        scale_26 = torch.conv2d(
            scale_25,
            l_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_25 = l_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_27 = torch.nn.functional.relu(scale_26, inplace=False)
        scale_26 = None
        scale_28 = torch.conv2d(
            scale_27,
            l_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_27 = l_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_13_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_29 = torch.nn.functional.hardsigmoid(scale_28, False)
        scale_28 = None
        input_108 = scale_29 * input_107
        scale_29 = input_107 = None
        input_109 = torch.conv2d(
            input_108,
            l_self_modules_backbone_modules_13_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_108 = l_self_modules_backbone_modules_13_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_110 = torch.nn.functional.batch_norm(
            input_109,
            l_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_109 = l_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_13_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_111 = torch.conv2d(
            input_110,
            l_self_modules_backbone_modules_14_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_14_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_112 = torch.nn.functional.batch_norm(
            input_111,
            l_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_111 = l_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_backbone_modules_14_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_113 = torch.nn.functional.hardswish(input_112, True)
        input_112 = None
        input_114 = torch.conv2d(
            input_113,
            l_self_modules_backbone_modules_14_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (2, 2),
            960,
        )
        input_113 = l_self_modules_backbone_modules_14_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_115 = torch.nn.functional.batch_norm(
            input_114,
            l_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_114 = l_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_14_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_116 = torch.nn.functional.hardswish(input_115, True)
        input_115 = None
        scale_30 = torch.nn.functional.adaptive_avg_pool2d(input_116, 1)
        scale_31 = torch.conv2d(
            scale_30,
            l_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_30 = l_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_32 = torch.nn.functional.relu(scale_31, inplace=False)
        scale_31 = None
        scale_33 = torch.conv2d(
            scale_32,
            l_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_32 = l_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_14_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_34 = torch.nn.functional.hardsigmoid(scale_33, False)
        scale_33 = None
        input_117 = scale_34 * input_116
        scale_34 = input_116 = None
        input_118 = torch.conv2d(
            input_117,
            l_self_modules_backbone_modules_14_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_117 = l_self_modules_backbone_modules_14_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_119 = torch.nn.functional.batch_norm(
            input_118,
            l_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_118 = l_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_14_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_119 += input_110
        result_8 = input_119
        input_119 = input_110 = None
        input_120 = torch.conv2d(
            result_8,
            l_self_modules_backbone_modules_15_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_15_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_121 = torch.nn.functional.batch_norm(
            input_120,
            l_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_120 = l_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_backbone_modules_15_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_122 = torch.nn.functional.hardswish(input_121, True)
        input_121 = None
        input_123 = torch.conv2d(
            input_122,
            l_self_modules_backbone_modules_15_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (4, 4),
            (2, 2),
            960,
        )
        input_122 = l_self_modules_backbone_modules_15_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_124 = torch.nn.functional.batch_norm(
            input_123,
            l_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_123 = l_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_backbone_modules_15_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_125 = torch.nn.functional.hardswish(input_124, True)
        input_124 = None
        scale_35 = torch.nn.functional.adaptive_avg_pool2d(input_125, 1)
        scale_36 = torch.conv2d(
            scale_35,
            l_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_35 = l_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_37 = torch.nn.functional.relu(scale_36, inplace=False)
        scale_36 = None
        scale_38 = torch.conv2d(
            scale_37,
            l_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_37 = l_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_backbone_modules_15_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_39 = torch.nn.functional.hardsigmoid(scale_38, False)
        scale_38 = None
        input_126 = scale_39 * input_125
        scale_39 = input_125 = None
        input_127 = torch.conv2d(
            input_126,
            l_self_modules_backbone_modules_15_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_126 = l_self_modules_backbone_modules_15_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_128 = torch.nn.functional.batch_norm(
            input_127,
            l_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_127 = l_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_backbone_modules_15_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_128 += result_8
        result_9 = input_128
        input_128 = result_8 = None
        input_129 = torch.conv2d(
            result_9,
            l_self_modules_backbone_modules_16_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_9 = (
            l_self_modules_backbone_modules_16_modules_0_parameters_weight_
        ) = None
        input_130 = torch.nn.functional.batch_norm(
            input_129,
            l_self_modules_backbone_modules_16_modules_1_buffers_running_mean_,
            l_self_modules_backbone_modules_16_modules_1_buffers_running_var_,
            l_self_modules_backbone_modules_16_modules_1_parameters_weight_,
            l_self_modules_backbone_modules_16_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_129 = (
            l_self_modules_backbone_modules_16_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_backbone_modules_16_modules_1_buffers_running_var_
        ) = (
            l_self_modules_backbone_modules_16_modules_1_parameters_weight_
        ) = l_self_modules_backbone_modules_16_modules_1_parameters_bias_ = None
        input_131 = torch.nn.functional.hardswish(input_130, True)
        input_130 = None
        input_132 = torch.conv2d(
            input_131,
            l_self_modules_classifier_modules_0_modules_convs_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_classifier_modules_0_modules_convs_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_133 = torch.nn.functional.batch_norm(
            input_132,
            l_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_buffers_running_var_,
            l_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_parameters_weight_,
            l_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_132 = l_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_buffers_running_mean_ = l_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_buffers_running_var_ = l_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_parameters_weight_ = l_self_modules_classifier_modules_0_modules_convs_modules_0_modules_1_parameters_bias_ = (None)
        input_134 = torch.nn.functional.relu(input_133, inplace=False)
        input_133 = None
        input_135 = torch.conv2d(
            input_131,
            l_self_modules_classifier_modules_0_modules_convs_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (12, 12),
            (12, 12),
            1,
        )
        l_self_modules_classifier_modules_0_modules_convs_modules_1_modules_0_parameters_weight_ = (
            None
        )
        input_136 = torch.nn.functional.batch_norm(
            input_135,
            l_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_buffers_running_var_,
            l_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_parameters_weight_,
            l_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_135 = l_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_buffers_running_mean_ = l_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_buffers_running_var_ = l_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_parameters_weight_ = l_self_modules_classifier_modules_0_modules_convs_modules_1_modules_1_parameters_bias_ = (None)
        input_137 = torch.nn.functional.relu(input_136, inplace=False)
        input_136 = None
        input_138 = torch.conv2d(
            input_131,
            l_self_modules_classifier_modules_0_modules_convs_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (24, 24),
            (24, 24),
            1,
        )
        l_self_modules_classifier_modules_0_modules_convs_modules_2_modules_0_parameters_weight_ = (
            None
        )
        input_139 = torch.nn.functional.batch_norm(
            input_138,
            l_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_buffers_running_var_,
            l_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_parameters_weight_,
            l_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_138 = l_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_buffers_running_mean_ = l_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_buffers_running_var_ = l_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_parameters_weight_ = l_self_modules_classifier_modules_0_modules_convs_modules_2_modules_1_parameters_bias_ = (None)
        input_140 = torch.nn.functional.relu(input_139, inplace=False)
        input_139 = None
        input_141 = torch.conv2d(
            input_131,
            l_self_modules_classifier_modules_0_modules_convs_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (36, 36),
            (36, 36),
            1,
        )
        l_self_modules_classifier_modules_0_modules_convs_modules_3_modules_0_parameters_weight_ = (
            None
        )
        input_142 = torch.nn.functional.batch_norm(
            input_141,
            l_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_buffers_running_var_,
            l_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_parameters_weight_,
            l_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_141 = l_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_buffers_running_mean_ = l_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_buffers_running_var_ = l_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_parameters_weight_ = l_self_modules_classifier_modules_0_modules_convs_modules_3_modules_1_parameters_bias_ = (None)
        input_143 = torch.nn.functional.relu(input_142, inplace=False)
        input_142 = None
        x = torch.nn.functional.adaptive_avg_pool2d(input_131, 1)
        input_131 = None
        x_1 = torch.conv2d(
            x,
            l_self_modules_classifier_modules_0_modules_convs_modules_4_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x = l_self_modules_classifier_modules_0_modules_convs_modules_4_modules_1_parameters_weight_ = (None)
        x_2 = torch.nn.functional.batch_norm(
            x_1,
            l_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_buffers_running_mean_,
            l_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_buffers_running_var_,
            l_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_parameters_weight_,
            l_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_1 = l_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_buffers_running_mean_ = l_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_buffers_running_var_ = l_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_parameters_weight_ = l_self_modules_classifier_modules_0_modules_convs_modules_4_modules_2_parameters_bias_ = (None)
        x_3 = torch.nn.functional.relu(x_2, inplace=False)
        x_2 = None
        interpolate = torch.nn.functional.interpolate(
            x_3, size=(14, 14), mode="bilinear", align_corners=False
        )
        x_3 = None
        res = torch.cat(
            [input_134, input_137, input_140, input_143, interpolate], dim=1
        )
        input_134 = input_137 = input_140 = input_143 = interpolate = None
        input_144 = torch.conv2d(
            res,
            l_self_modules_classifier_modules_0_modules_project_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        res = l_self_modules_classifier_modules_0_modules_project_modules_0_parameters_weight_ = (None)
        input_145 = torch.nn.functional.batch_norm(
            input_144,
            l_self_modules_classifier_modules_0_modules_project_modules_1_buffers_running_mean_,
            l_self_modules_classifier_modules_0_modules_project_modules_1_buffers_running_var_,
            l_self_modules_classifier_modules_0_modules_project_modules_1_parameters_weight_,
            l_self_modules_classifier_modules_0_modules_project_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_144 = l_self_modules_classifier_modules_0_modules_project_modules_1_buffers_running_mean_ = l_self_modules_classifier_modules_0_modules_project_modules_1_buffers_running_var_ = l_self_modules_classifier_modules_0_modules_project_modules_1_parameters_weight_ = l_self_modules_classifier_modules_0_modules_project_modules_1_parameters_bias_ = (None)
        input_146 = torch.nn.functional.relu(input_145, inplace=False)
        input_145 = None
        input_147 = torch.nn.functional.dropout(input_146, 0.5, False, False)
        input_146 = None
        input_148 = torch.conv2d(
            input_147,
            l_self_modules_classifier_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_147 = l_self_modules_classifier_modules_1_parameters_weight_ = None
        input_149 = torch.nn.functional.batch_norm(
            input_148,
            l_self_modules_classifier_modules_2_buffers_running_mean_,
            l_self_modules_classifier_modules_2_buffers_running_var_,
            l_self_modules_classifier_modules_2_parameters_weight_,
            l_self_modules_classifier_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_148 = (
            l_self_modules_classifier_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_classifier_modules_2_buffers_running_var_
        ) = (
            l_self_modules_classifier_modules_2_parameters_weight_
        ) = l_self_modules_classifier_modules_2_parameters_bias_ = None
        input_150 = torch.nn.functional.relu(input_149, inplace=False)
        input_149 = None
        input_151 = torch.conv2d(
            input_150,
            l_self_modules_classifier_modules_4_parameters_weight_,
            l_self_modules_classifier_modules_4_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_150 = (
            l_self_modules_classifier_modules_4_parameters_weight_
        ) = l_self_modules_classifier_modules_4_parameters_bias_ = None
        x_4 = torch.nn.functional.interpolate(
            input_151, size=(224, 224), mode="bilinear", align_corners=False
        )
        input_151 = None
        input_152 = torch.conv2d(
            input_33,
            l_self_modules_aux_classifier_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_33 = l_self_modules_aux_classifier_modules_0_parameters_weight_ = None
        input_153 = torch.nn.functional.batch_norm(
            input_152,
            l_self_modules_aux_classifier_modules_1_buffers_running_mean_,
            l_self_modules_aux_classifier_modules_1_buffers_running_var_,
            l_self_modules_aux_classifier_modules_1_parameters_weight_,
            l_self_modules_aux_classifier_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_152 = (
            l_self_modules_aux_classifier_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_aux_classifier_modules_1_buffers_running_var_
        ) = (
            l_self_modules_aux_classifier_modules_1_parameters_weight_
        ) = l_self_modules_aux_classifier_modules_1_parameters_bias_ = None
        input_154 = torch.nn.functional.relu(input_153, inplace=False)
        input_153 = None
        input_155 = torch.nn.functional.dropout(input_154, 0.1, False, False)
        input_154 = None
        input_156 = torch.conv2d(
            input_155,
            l_self_modules_aux_classifier_modules_4_parameters_weight_,
            l_self_modules_aux_classifier_modules_4_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_155 = (
            l_self_modules_aux_classifier_modules_4_parameters_weight_
        ) = l_self_modules_aux_classifier_modules_4_parameters_bias_ = None
        x_5 = torch.nn.functional.interpolate(
            input_156, size=(224, 224), mode="bilinear", align_corners=False
        )
        input_156 = None
        return (x_4, x_5)
