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
        L_self_modules_features_modules_1_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_block_modules_1_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_block_modules_1_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_block_modules_1_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_block_modules_1_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_block_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_block_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_1_modules_block_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_1_modules_block_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_1_modules_block_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_block_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_block_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_2_modules_block_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_2_modules_block_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_2_modules_block_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_block_modules_2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_block_modules_2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_3_modules_block_modules_2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_3_modules_block_modules_2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_3_modules_block_modules_2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_4_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_4_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_4_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_5_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_5_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_5_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_6_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_6_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_6_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_7_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_7_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_7_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_8_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_8_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_8_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_9_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_9_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_9_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_9_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_9_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_9_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_9_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_10_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_10_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_10_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_10_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_10_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_10_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_10_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_block_modules_0_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_block_modules_0_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_block_modules_0_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_block_modules_0_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_block_modules_1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_block_modules_1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_block_modules_1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_block_modules_1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_block_modules_3_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_11_modules_block_modules_3_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_11_modules_block_modules_3_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_11_modules_block_modules_3_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_features_modules_12_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_features_modules_12_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_features_modules_12_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classifier_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_features_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_1_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_1_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_1_modules_block_modules_1_modules_fc1_parameters_weight_ = L_self_modules_features_modules_1_modules_block_modules_1_modules_fc1_parameters_weight_
        l_self_modules_features_modules_1_modules_block_modules_1_modules_fc1_parameters_bias_ = L_self_modules_features_modules_1_modules_block_modules_1_modules_fc1_parameters_bias_
        l_self_modules_features_modules_1_modules_block_modules_1_modules_fc2_parameters_weight_ = L_self_modules_features_modules_1_modules_block_modules_1_modules_fc2_parameters_weight_
        l_self_modules_features_modules_1_modules_block_modules_1_modules_fc2_parameters_bias_ = L_self_modules_features_modules_1_modules_block_modules_1_modules_fc2_parameters_bias_
        l_self_modules_features_modules_1_modules_block_modules_2_modules_0_parameters_weight_ = L_self_modules_features_modules_1_modules_block_modules_2_modules_0_parameters_weight_
        l_self_modules_features_modules_1_modules_block_modules_2_modules_1_buffers_running_mean_ = L_self_modules_features_modules_1_modules_block_modules_2_modules_1_buffers_running_mean_
        l_self_modules_features_modules_1_modules_block_modules_2_modules_1_buffers_running_var_ = L_self_modules_features_modules_1_modules_block_modules_2_modules_1_buffers_running_var_
        l_self_modules_features_modules_1_modules_block_modules_2_modules_1_parameters_weight_ = L_self_modules_features_modules_1_modules_block_modules_2_modules_1_parameters_weight_
        l_self_modules_features_modules_1_modules_block_modules_2_modules_1_parameters_bias_ = L_self_modules_features_modules_1_modules_block_modules_2_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_2_modules_block_modules_2_modules_0_parameters_weight_ = L_self_modules_features_modules_2_modules_block_modules_2_modules_0_parameters_weight_
        l_self_modules_features_modules_2_modules_block_modules_2_modules_1_buffers_running_mean_ = L_self_modules_features_modules_2_modules_block_modules_2_modules_1_buffers_running_mean_
        l_self_modules_features_modules_2_modules_block_modules_2_modules_1_buffers_running_var_ = L_self_modules_features_modules_2_modules_block_modules_2_modules_1_buffers_running_var_
        l_self_modules_features_modules_2_modules_block_modules_2_modules_1_parameters_weight_ = L_self_modules_features_modules_2_modules_block_modules_2_modules_1_parameters_weight_
        l_self_modules_features_modules_2_modules_block_modules_2_modules_1_parameters_bias_ = L_self_modules_features_modules_2_modules_block_modules_2_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_3_modules_block_modules_2_modules_0_parameters_weight_ = L_self_modules_features_modules_3_modules_block_modules_2_modules_0_parameters_weight_
        l_self_modules_features_modules_3_modules_block_modules_2_modules_1_buffers_running_mean_ = L_self_modules_features_modules_3_modules_block_modules_2_modules_1_buffers_running_mean_
        l_self_modules_features_modules_3_modules_block_modules_2_modules_1_buffers_running_var_ = L_self_modules_features_modules_3_modules_block_modules_2_modules_1_buffers_running_var_
        l_self_modules_features_modules_3_modules_block_modules_2_modules_1_parameters_weight_ = L_self_modules_features_modules_3_modules_block_modules_2_modules_1_parameters_weight_
        l_self_modules_features_modules_3_modules_block_modules_2_modules_1_parameters_bias_ = L_self_modules_features_modules_3_modules_block_modules_2_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_4_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_4_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_4_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_4_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_5_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_5_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_5_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_5_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_6_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_6_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_6_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_6_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_7_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_7_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_7_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_7_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_7_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_7_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_7_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_7_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_8_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_8_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_8_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_8_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_8_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_8_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_8_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_8_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_8_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_8_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_8_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_8_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_8_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_8_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_8_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_8_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_8_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_8_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_8_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_8_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_8_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_8_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_8_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_8_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_9_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_9_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_9_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_9_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_9_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_9_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_9_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_9_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_9_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_9_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_9_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_9_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_9_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_9_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_9_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_9_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_9_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_9_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_9_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_9_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_9_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_9_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_9_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_9_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_10_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_10_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_10_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_10_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_10_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_10_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_10_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_10_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_10_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_10_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_10_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_10_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_10_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_10_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_10_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_10_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_10_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_10_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_10_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_10_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_10_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_10_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_10_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_10_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_11_modules_block_modules_0_modules_0_parameters_weight_ = L_self_modules_features_modules_11_modules_block_modules_0_modules_0_parameters_weight_
        l_self_modules_features_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_ = L_self_modules_features_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_
        l_self_modules_features_modules_11_modules_block_modules_0_modules_1_buffers_running_var_ = L_self_modules_features_modules_11_modules_block_modules_0_modules_1_buffers_running_var_
        l_self_modules_features_modules_11_modules_block_modules_0_modules_1_parameters_weight_ = L_self_modules_features_modules_11_modules_block_modules_0_modules_1_parameters_weight_
        l_self_modules_features_modules_11_modules_block_modules_0_modules_1_parameters_bias_ = L_self_modules_features_modules_11_modules_block_modules_0_modules_1_parameters_bias_
        l_self_modules_features_modules_11_modules_block_modules_1_modules_0_parameters_weight_ = L_self_modules_features_modules_11_modules_block_modules_1_modules_0_parameters_weight_
        l_self_modules_features_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_ = L_self_modules_features_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_
        l_self_modules_features_modules_11_modules_block_modules_1_modules_1_buffers_running_var_ = L_self_modules_features_modules_11_modules_block_modules_1_modules_1_buffers_running_var_
        l_self_modules_features_modules_11_modules_block_modules_1_modules_1_parameters_weight_ = L_self_modules_features_modules_11_modules_block_modules_1_modules_1_parameters_weight_
        l_self_modules_features_modules_11_modules_block_modules_1_modules_1_parameters_bias_ = L_self_modules_features_modules_11_modules_block_modules_1_modules_1_parameters_bias_
        l_self_modules_features_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_ = L_self_modules_features_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_
        l_self_modules_features_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_ = L_self_modules_features_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_
        l_self_modules_features_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_ = L_self_modules_features_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_
        l_self_modules_features_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_ = L_self_modules_features_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_
        l_self_modules_features_modules_11_modules_block_modules_3_modules_0_parameters_weight_ = L_self_modules_features_modules_11_modules_block_modules_3_modules_0_parameters_weight_
        l_self_modules_features_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_ = L_self_modules_features_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_
        l_self_modules_features_modules_11_modules_block_modules_3_modules_1_buffers_running_var_ = L_self_modules_features_modules_11_modules_block_modules_3_modules_1_buffers_running_var_
        l_self_modules_features_modules_11_modules_block_modules_3_modules_1_parameters_weight_ = L_self_modules_features_modules_11_modules_block_modules_3_modules_1_parameters_weight_
        l_self_modules_features_modules_11_modules_block_modules_3_modules_1_parameters_bias_ = L_self_modules_features_modules_11_modules_block_modules_3_modules_1_parameters_bias_
        l_self_modules_features_modules_12_modules_0_parameters_weight_ = (
            L_self_modules_features_modules_12_modules_0_parameters_weight_
        )
        l_self_modules_features_modules_12_modules_1_buffers_running_mean_ = (
            L_self_modules_features_modules_12_modules_1_buffers_running_mean_
        )
        l_self_modules_features_modules_12_modules_1_buffers_running_var_ = (
            L_self_modules_features_modules_12_modules_1_buffers_running_var_
        )
        l_self_modules_features_modules_12_modules_1_parameters_weight_ = (
            L_self_modules_features_modules_12_modules_1_parameters_weight_
        )
        l_self_modules_features_modules_12_modules_1_parameters_bias_ = (
            L_self_modules_features_modules_12_modules_1_parameters_bias_
        )
        l_self_modules_classifier_modules_0_parameters_weight_ = (
            L_self_modules_classifier_modules_0_parameters_weight_
        )
        l_self_modules_classifier_modules_0_parameters_bias_ = (
            L_self_modules_classifier_modules_0_parameters_bias_
        )
        l_self_modules_classifier_modules_3_parameters_weight_ = (
            L_self_modules_classifier_modules_3_parameters_weight_
        )
        l_self_modules_classifier_modules_3_parameters_bias_ = (
            L_self_modules_classifier_modules_3_parameters_bias_
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
        input_3 = torch.nn.functional.hardswish(input_2, True)
        input_2 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_features_modules_1_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            16,
        )
        input_3 = l_self_modules_features_modules_1_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_features_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_4 = l_self_modules_features_modules_1_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_1_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_1_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_6 = torch.nn.functional.relu(input_5, inplace=True)
        input_5 = None
        scale = torch.nn.functional.adaptive_avg_pool2d(input_6, 1)
        scale_1 = torch.conv2d(
            scale,
            l_self_modules_features_modules_1_modules_block_modules_1_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_1_modules_block_modules_1_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale = l_self_modules_features_modules_1_modules_block_modules_1_modules_fc1_parameters_weight_ = l_self_modules_features_modules_1_modules_block_modules_1_modules_fc1_parameters_bias_ = (None)
        scale_2 = torch.nn.functional.relu(scale_1, inplace=False)
        scale_1 = None
        scale_3 = torch.conv2d(
            scale_2,
            l_self_modules_features_modules_1_modules_block_modules_1_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_1_modules_block_modules_1_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_2 = l_self_modules_features_modules_1_modules_block_modules_1_modules_fc2_parameters_weight_ = l_self_modules_features_modules_1_modules_block_modules_1_modules_fc2_parameters_bias_ = (None)
        scale_4 = torch.nn.functional.hardsigmoid(scale_3, False)
        scale_3 = None
        input_7 = scale_4 * input_6
        scale_4 = input_6 = None
        input_8 = torch.conv2d(
            input_7,
            l_self_modules_features_modules_1_modules_block_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_7 = l_self_modules_features_modules_1_modules_block_modules_2_modules_0_parameters_weight_ = (None)
        input_9 = torch.nn.functional.batch_norm(
            input_8,
            l_self_modules_features_modules_1_modules_block_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_1_modules_block_modules_2_modules_1_buffers_running_var_,
            l_self_modules_features_modules_1_modules_block_modules_2_modules_1_parameters_weight_,
            l_self_modules_features_modules_1_modules_block_modules_2_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_8 = l_self_modules_features_modules_1_modules_block_modules_2_modules_1_buffers_running_mean_ = l_self_modules_features_modules_1_modules_block_modules_2_modules_1_buffers_running_var_ = l_self_modules_features_modules_1_modules_block_modules_2_modules_1_parameters_weight_ = l_self_modules_features_modules_1_modules_block_modules_2_modules_1_parameters_bias_ = (None)
        input_10 = torch.conv2d(
            input_9,
            l_self_modules_features_modules_2_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_9 = l_self_modules_features_modules_2_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_11 = torch.nn.functional.batch_norm(
            input_10,
            l_self_modules_features_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_10 = l_self_modules_features_modules_2_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_12 = torch.nn.functional.relu(input_11, inplace=True)
        input_11 = None
        input_13 = torch.conv2d(
            input_12,
            l_self_modules_features_modules_2_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            72,
        )
        input_12 = l_self_modules_features_modules_2_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_14 = torch.nn.functional.batch_norm(
            input_13,
            l_self_modules_features_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_13 = l_self_modules_features_modules_2_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_15 = torch.nn.functional.relu(input_14, inplace=True)
        input_14 = None
        input_16 = torch.conv2d(
            input_15,
            l_self_modules_features_modules_2_modules_block_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_15 = l_self_modules_features_modules_2_modules_block_modules_2_modules_0_parameters_weight_ = (None)
        input_17 = torch.nn.functional.batch_norm(
            input_16,
            l_self_modules_features_modules_2_modules_block_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_2_modules_block_modules_2_modules_1_buffers_running_var_,
            l_self_modules_features_modules_2_modules_block_modules_2_modules_1_parameters_weight_,
            l_self_modules_features_modules_2_modules_block_modules_2_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_16 = l_self_modules_features_modules_2_modules_block_modules_2_modules_1_buffers_running_mean_ = l_self_modules_features_modules_2_modules_block_modules_2_modules_1_buffers_running_var_ = l_self_modules_features_modules_2_modules_block_modules_2_modules_1_parameters_weight_ = l_self_modules_features_modules_2_modules_block_modules_2_modules_1_parameters_bias_ = (None)
        input_18 = torch.conv2d(
            input_17,
            l_self_modules_features_modules_3_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_3_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_19 = torch.nn.functional.batch_norm(
            input_18,
            l_self_modules_features_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_18 = l_self_modules_features_modules_3_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_20 = torch.nn.functional.relu(input_19, inplace=True)
        input_19 = None
        input_21 = torch.conv2d(
            input_20,
            l_self_modules_features_modules_3_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            88,
        )
        input_20 = l_self_modules_features_modules_3_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_22 = torch.nn.functional.batch_norm(
            input_21,
            l_self_modules_features_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_21 = l_self_modules_features_modules_3_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_23 = torch.nn.functional.relu(input_22, inplace=True)
        input_22 = None
        input_24 = torch.conv2d(
            input_23,
            l_self_modules_features_modules_3_modules_block_modules_2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_23 = l_self_modules_features_modules_3_modules_block_modules_2_modules_0_parameters_weight_ = (None)
        input_25 = torch.nn.functional.batch_norm(
            input_24,
            l_self_modules_features_modules_3_modules_block_modules_2_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_3_modules_block_modules_2_modules_1_buffers_running_var_,
            l_self_modules_features_modules_3_modules_block_modules_2_modules_1_parameters_weight_,
            l_self_modules_features_modules_3_modules_block_modules_2_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_24 = l_self_modules_features_modules_3_modules_block_modules_2_modules_1_buffers_running_mean_ = l_self_modules_features_modules_3_modules_block_modules_2_modules_1_buffers_running_var_ = l_self_modules_features_modules_3_modules_block_modules_2_modules_1_parameters_weight_ = l_self_modules_features_modules_3_modules_block_modules_2_modules_1_parameters_bias_ = (None)
        input_25 += input_17
        result = input_25
        input_25 = input_17 = None
        input_26 = torch.conv2d(
            result,
            l_self_modules_features_modules_4_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result = l_self_modules_features_modules_4_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_27 = torch.nn.functional.batch_norm(
            input_26,
            l_self_modules_features_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_26 = l_self_modules_features_modules_4_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_28 = torch.nn.functional.hardswish(input_27, True)
        input_27 = None
        input_29 = torch.conv2d(
            input_28,
            l_self_modules_features_modules_4_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            96,
        )
        input_28 = l_self_modules_features_modules_4_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_30 = torch.nn.functional.batch_norm(
            input_29,
            l_self_modules_features_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_29 = l_self_modules_features_modules_4_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_31 = torch.nn.functional.hardswish(input_30, True)
        input_30 = None
        scale_5 = torch.nn.functional.adaptive_avg_pool2d(input_31, 1)
        scale_6 = torch.conv2d(
            scale_5,
            l_self_modules_features_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_5 = l_self_modules_features_modules_4_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_4_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_7 = torch.nn.functional.relu(scale_6, inplace=False)
        scale_6 = None
        scale_8 = torch.conv2d(
            scale_7,
            l_self_modules_features_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_7 = l_self_modules_features_modules_4_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_4_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_9 = torch.nn.functional.hardsigmoid(scale_8, False)
        scale_8 = None
        input_32 = scale_9 * input_31
        scale_9 = input_31 = None
        input_33 = torch.conv2d(
            input_32,
            l_self_modules_features_modules_4_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_32 = l_self_modules_features_modules_4_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_34 = torch.nn.functional.batch_norm(
            input_33,
            l_self_modules_features_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_4_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_4_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_4_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_33 = l_self_modules_features_modules_4_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_4_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_4_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_4_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_35 = torch.conv2d(
            input_34,
            l_self_modules_features_modules_5_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_5_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_36 = torch.nn.functional.batch_norm(
            input_35,
            l_self_modules_features_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_35 = l_self_modules_features_modules_5_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_37 = torch.nn.functional.hardswish(input_36, True)
        input_36 = None
        input_38 = torch.conv2d(
            input_37,
            l_self_modules_features_modules_5_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            240,
        )
        input_37 = l_self_modules_features_modules_5_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_39 = torch.nn.functional.batch_norm(
            input_38,
            l_self_modules_features_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_38 = l_self_modules_features_modules_5_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_40 = torch.nn.functional.hardswish(input_39, True)
        input_39 = None
        scale_10 = torch.nn.functional.adaptive_avg_pool2d(input_40, 1)
        scale_11 = torch.conv2d(
            scale_10,
            l_self_modules_features_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_10 = l_self_modules_features_modules_5_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_5_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_12 = torch.nn.functional.relu(scale_11, inplace=False)
        scale_11 = None
        scale_13 = torch.conv2d(
            scale_12,
            l_self_modules_features_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_12 = l_self_modules_features_modules_5_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_5_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_14 = torch.nn.functional.hardsigmoid(scale_13, False)
        scale_13 = None
        input_41 = scale_14 * input_40
        scale_14 = input_40 = None
        input_42 = torch.conv2d(
            input_41,
            l_self_modules_features_modules_5_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_41 = l_self_modules_features_modules_5_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_43 = torch.nn.functional.batch_norm(
            input_42,
            l_self_modules_features_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_5_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_5_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_5_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_42 = l_self_modules_features_modules_5_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_5_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_5_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_5_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_43 += input_34
        result_1 = input_43
        input_43 = input_34 = None
        input_44 = torch.conv2d(
            result_1,
            l_self_modules_features_modules_6_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_6_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_45 = torch.nn.functional.batch_norm(
            input_44,
            l_self_modules_features_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_44 = l_self_modules_features_modules_6_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_46 = torch.nn.functional.hardswish(input_45, True)
        input_45 = None
        input_47 = torch.conv2d(
            input_46,
            l_self_modules_features_modules_6_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            240,
        )
        input_46 = l_self_modules_features_modules_6_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_48 = torch.nn.functional.batch_norm(
            input_47,
            l_self_modules_features_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_47 = l_self_modules_features_modules_6_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_49 = torch.nn.functional.hardswish(input_48, True)
        input_48 = None
        scale_15 = torch.nn.functional.adaptive_avg_pool2d(input_49, 1)
        scale_16 = torch.conv2d(
            scale_15,
            l_self_modules_features_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_15 = l_self_modules_features_modules_6_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_6_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_17 = torch.nn.functional.relu(scale_16, inplace=False)
        scale_16 = None
        scale_18 = torch.conv2d(
            scale_17,
            l_self_modules_features_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_17 = l_self_modules_features_modules_6_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_6_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_19 = torch.nn.functional.hardsigmoid(scale_18, False)
        scale_18 = None
        input_50 = scale_19 * input_49
        scale_19 = input_49 = None
        input_51 = torch.conv2d(
            input_50,
            l_self_modules_features_modules_6_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_50 = l_self_modules_features_modules_6_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_52 = torch.nn.functional.batch_norm(
            input_51,
            l_self_modules_features_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_6_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_6_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_6_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_51 = l_self_modules_features_modules_6_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_6_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_6_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_6_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_52 += result_1
        result_2 = input_52
        input_52 = result_1 = None
        input_53 = torch.conv2d(
            result_2,
            l_self_modules_features_modules_7_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_2 = l_self_modules_features_modules_7_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_54 = torch.nn.functional.batch_norm(
            input_53,
            l_self_modules_features_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_53 = l_self_modules_features_modules_7_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_55 = torch.nn.functional.hardswish(input_54, True)
        input_54 = None
        input_56 = torch.conv2d(
            input_55,
            l_self_modules_features_modules_7_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            120,
        )
        input_55 = l_self_modules_features_modules_7_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_57 = torch.nn.functional.batch_norm(
            input_56,
            l_self_modules_features_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_56 = l_self_modules_features_modules_7_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_58 = torch.nn.functional.hardswish(input_57, True)
        input_57 = None
        scale_20 = torch.nn.functional.adaptive_avg_pool2d(input_58, 1)
        scale_21 = torch.conv2d(
            scale_20,
            l_self_modules_features_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_20 = l_self_modules_features_modules_7_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_7_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_22 = torch.nn.functional.relu(scale_21, inplace=False)
        scale_21 = None
        scale_23 = torch.conv2d(
            scale_22,
            l_self_modules_features_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_22 = l_self_modules_features_modules_7_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_7_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_24 = torch.nn.functional.hardsigmoid(scale_23, False)
        scale_23 = None
        input_59 = scale_24 * input_58
        scale_24 = input_58 = None
        input_60 = torch.conv2d(
            input_59,
            l_self_modules_features_modules_7_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_59 = l_self_modules_features_modules_7_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_61 = torch.nn.functional.batch_norm(
            input_60,
            l_self_modules_features_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_7_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_7_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_7_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_60 = l_self_modules_features_modules_7_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_7_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_7_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_7_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_62 = torch.conv2d(
            input_61,
            l_self_modules_features_modules_8_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_8_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_63 = torch.nn.functional.batch_norm(
            input_62,
            l_self_modules_features_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_8_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_8_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_62 = l_self_modules_features_modules_8_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_8_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_8_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_8_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_64 = torch.nn.functional.hardswish(input_63, True)
        input_63 = None
        input_65 = torch.conv2d(
            input_64,
            l_self_modules_features_modules_8_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            144,
        )
        input_64 = l_self_modules_features_modules_8_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_66 = torch.nn.functional.batch_norm(
            input_65,
            l_self_modules_features_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_8_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_8_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_65 = l_self_modules_features_modules_8_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_8_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_8_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_8_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_67 = torch.nn.functional.hardswish(input_66, True)
        input_66 = None
        scale_25 = torch.nn.functional.adaptive_avg_pool2d(input_67, 1)
        scale_26 = torch.conv2d(
            scale_25,
            l_self_modules_features_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_25 = l_self_modules_features_modules_8_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_8_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_27 = torch.nn.functional.relu(scale_26, inplace=False)
        scale_26 = None
        scale_28 = torch.conv2d(
            scale_27,
            l_self_modules_features_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_27 = l_self_modules_features_modules_8_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_8_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_29 = torch.nn.functional.hardsigmoid(scale_28, False)
        scale_28 = None
        input_68 = scale_29 * input_67
        scale_29 = input_67 = None
        input_69 = torch.conv2d(
            input_68,
            l_self_modules_features_modules_8_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_68 = l_self_modules_features_modules_8_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_70 = torch.nn.functional.batch_norm(
            input_69,
            l_self_modules_features_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_8_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_8_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_8_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_69 = l_self_modules_features_modules_8_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_8_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_8_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_8_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_70 += input_61
        result_3 = input_70
        input_70 = input_61 = None
        input_71 = torch.conv2d(
            result_3,
            l_self_modules_features_modules_9_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_3 = l_self_modules_features_modules_9_modules_block_modules_0_modules_0_parameters_weight_ = (None)
        input_72 = torch.nn.functional.batch_norm(
            input_71,
            l_self_modules_features_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_9_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_9_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_9_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_71 = l_self_modules_features_modules_9_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_9_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_9_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_9_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_73 = torch.nn.functional.hardswish(input_72, True)
        input_72 = None
        input_74 = torch.conv2d(
            input_73,
            l_self_modules_features_modules_9_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (1, 1),
            288,
        )
        input_73 = l_self_modules_features_modules_9_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_75 = torch.nn.functional.batch_norm(
            input_74,
            l_self_modules_features_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_9_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_9_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_9_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_74 = l_self_modules_features_modules_9_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_9_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_9_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_9_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_76 = torch.nn.functional.hardswish(input_75, True)
        input_75 = None
        scale_30 = torch.nn.functional.adaptive_avg_pool2d(input_76, 1)
        scale_31 = torch.conv2d(
            scale_30,
            l_self_modules_features_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_30 = l_self_modules_features_modules_9_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_9_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_32 = torch.nn.functional.relu(scale_31, inplace=False)
        scale_31 = None
        scale_33 = torch.conv2d(
            scale_32,
            l_self_modules_features_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_32 = l_self_modules_features_modules_9_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_9_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_34 = torch.nn.functional.hardsigmoid(scale_33, False)
        scale_33 = None
        input_77 = scale_34 * input_76
        scale_34 = input_76 = None
        input_78 = torch.conv2d(
            input_77,
            l_self_modules_features_modules_9_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_77 = l_self_modules_features_modules_9_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_79 = torch.nn.functional.batch_norm(
            input_78,
            l_self_modules_features_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_9_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_9_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_9_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_78 = l_self_modules_features_modules_9_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_9_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_9_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_9_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_80 = torch.conv2d(
            input_79,
            l_self_modules_features_modules_10_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_10_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_81 = torch.nn.functional.batch_norm(
            input_80,
            l_self_modules_features_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_10_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_10_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_10_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_80 = l_self_modules_features_modules_10_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_10_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_10_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_10_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_82 = torch.nn.functional.hardswish(input_81, True)
        input_81 = None
        input_83 = torch.conv2d(
            input_82,
            l_self_modules_features_modules_10_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            576,
        )
        input_82 = l_self_modules_features_modules_10_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_84 = torch.nn.functional.batch_norm(
            input_83,
            l_self_modules_features_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_10_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_10_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_10_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_83 = l_self_modules_features_modules_10_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_10_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_10_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_10_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_85 = torch.nn.functional.hardswish(input_84, True)
        input_84 = None
        scale_35 = torch.nn.functional.adaptive_avg_pool2d(input_85, 1)
        scale_36 = torch.conv2d(
            scale_35,
            l_self_modules_features_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_35 = l_self_modules_features_modules_10_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_10_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_37 = torch.nn.functional.relu(scale_36, inplace=False)
        scale_36 = None
        scale_38 = torch.conv2d(
            scale_37,
            l_self_modules_features_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_37 = l_self_modules_features_modules_10_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_10_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_39 = torch.nn.functional.hardsigmoid(scale_38, False)
        scale_38 = None
        input_86 = scale_39 * input_85
        scale_39 = input_85 = None
        input_87 = torch.conv2d(
            input_86,
            l_self_modules_features_modules_10_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_86 = l_self_modules_features_modules_10_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_88 = torch.nn.functional.batch_norm(
            input_87,
            l_self_modules_features_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_10_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_10_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_10_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_87 = l_self_modules_features_modules_10_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_10_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_10_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_10_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_88 += input_79
        result_4 = input_88
        input_88 = input_79 = None
        input_89 = torch.conv2d(
            result_4,
            l_self_modules_features_modules_11_modules_block_modules_0_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_features_modules_11_modules_block_modules_0_modules_0_parameters_weight_ = (
            None
        )
        input_90 = torch.nn.functional.batch_norm(
            input_89,
            l_self_modules_features_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_block_modules_0_modules_1_buffers_running_var_,
            l_self_modules_features_modules_11_modules_block_modules_0_modules_1_parameters_weight_,
            l_self_modules_features_modules_11_modules_block_modules_0_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_89 = l_self_modules_features_modules_11_modules_block_modules_0_modules_1_buffers_running_mean_ = l_self_modules_features_modules_11_modules_block_modules_0_modules_1_buffers_running_var_ = l_self_modules_features_modules_11_modules_block_modules_0_modules_1_parameters_weight_ = l_self_modules_features_modules_11_modules_block_modules_0_modules_1_parameters_bias_ = (None)
        input_91 = torch.nn.functional.hardswish(input_90, True)
        input_90 = None
        input_92 = torch.conv2d(
            input_91,
            l_self_modules_features_modules_11_modules_block_modules_1_modules_0_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            576,
        )
        input_91 = l_self_modules_features_modules_11_modules_block_modules_1_modules_0_parameters_weight_ = (None)
        input_93 = torch.nn.functional.batch_norm(
            input_92,
            l_self_modules_features_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_block_modules_1_modules_1_buffers_running_var_,
            l_self_modules_features_modules_11_modules_block_modules_1_modules_1_parameters_weight_,
            l_self_modules_features_modules_11_modules_block_modules_1_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_92 = l_self_modules_features_modules_11_modules_block_modules_1_modules_1_buffers_running_mean_ = l_self_modules_features_modules_11_modules_block_modules_1_modules_1_buffers_running_var_ = l_self_modules_features_modules_11_modules_block_modules_1_modules_1_parameters_weight_ = l_self_modules_features_modules_11_modules_block_modules_1_modules_1_parameters_bias_ = (None)
        input_94 = torch.nn.functional.hardswish(input_93, True)
        input_93 = None
        scale_40 = torch.nn.functional.adaptive_avg_pool2d(input_94, 1)
        scale_41 = torch.conv2d(
            scale_40,
            l_self_modules_features_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_,
            l_self_modules_features_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_40 = l_self_modules_features_modules_11_modules_block_modules_2_modules_fc1_parameters_weight_ = l_self_modules_features_modules_11_modules_block_modules_2_modules_fc1_parameters_bias_ = (None)
        scale_42 = torch.nn.functional.relu(scale_41, inplace=False)
        scale_41 = None
        scale_43 = torch.conv2d(
            scale_42,
            l_self_modules_features_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_,
            l_self_modules_features_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        scale_42 = l_self_modules_features_modules_11_modules_block_modules_2_modules_fc2_parameters_weight_ = l_self_modules_features_modules_11_modules_block_modules_2_modules_fc2_parameters_bias_ = (None)
        scale_44 = torch.nn.functional.hardsigmoid(scale_43, False)
        scale_43 = None
        input_95 = scale_44 * input_94
        scale_44 = input_94 = None
        input_96 = torch.conv2d(
            input_95,
            l_self_modules_features_modules_11_modules_block_modules_3_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_95 = l_self_modules_features_modules_11_modules_block_modules_3_modules_0_parameters_weight_ = (None)
        input_97 = torch.nn.functional.batch_norm(
            input_96,
            l_self_modules_features_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_11_modules_block_modules_3_modules_1_buffers_running_var_,
            l_self_modules_features_modules_11_modules_block_modules_3_modules_1_parameters_weight_,
            l_self_modules_features_modules_11_modules_block_modules_3_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_96 = l_self_modules_features_modules_11_modules_block_modules_3_modules_1_buffers_running_mean_ = l_self_modules_features_modules_11_modules_block_modules_3_modules_1_buffers_running_var_ = l_self_modules_features_modules_11_modules_block_modules_3_modules_1_parameters_weight_ = l_self_modules_features_modules_11_modules_block_modules_3_modules_1_parameters_bias_ = (None)
        input_97 += result_4
        result_5 = input_97
        input_97 = result_4 = None
        input_98 = torch.conv2d(
            result_5,
            l_self_modules_features_modules_12_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        result_5 = (
            l_self_modules_features_modules_12_modules_0_parameters_weight_
        ) = None
        input_99 = torch.nn.functional.batch_norm(
            input_98,
            l_self_modules_features_modules_12_modules_1_buffers_running_mean_,
            l_self_modules_features_modules_12_modules_1_buffers_running_var_,
            l_self_modules_features_modules_12_modules_1_parameters_weight_,
            l_self_modules_features_modules_12_modules_1_parameters_bias_,
            False,
            0.01,
            0.001,
        )
        input_98 = (
            l_self_modules_features_modules_12_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_features_modules_12_modules_1_buffers_running_var_
        ) = (
            l_self_modules_features_modules_12_modules_1_parameters_weight_
        ) = l_self_modules_features_modules_12_modules_1_parameters_bias_ = None
        input_100 = torch.nn.functional.hardswish(input_99, True)
        input_99 = None
        x = torch.nn.functional.adaptive_avg_pool2d(input_100, 1)
        input_100 = None
        x_1 = torch.flatten(x, 1)
        x = None
        input_101 = torch._C._nn.linear(
            x_1,
            l_self_modules_classifier_modules_0_parameters_weight_,
            l_self_modules_classifier_modules_0_parameters_bias_,
        )
        x_1 = (
            l_self_modules_classifier_modules_0_parameters_weight_
        ) = l_self_modules_classifier_modules_0_parameters_bias_ = None
        input_102 = torch.nn.functional.hardswish(input_101, True)
        input_101 = None
        input_103 = torch.nn.functional.dropout(input_102, 0.2, False, True)
        input_102 = None
        input_104 = torch._C._nn.linear(
            input_103,
            l_self_modules_classifier_modules_3_parameters_weight_,
            l_self_modules_classifier_modules_3_parameters_bias_,
        )
        input_103 = (
            l_self_modules_classifier_modules_3_parameters_weight_
        ) = l_self_modules_classifier_modules_3_parameters_bias_ = None
        return (input_104,)
