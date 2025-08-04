import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_conv1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_x_: torch.Tensor,
        L_self_modules_conv1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_conv1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branch1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branch1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branch1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branch1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branch1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branch1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branch1_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branch1_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branch1_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branch1_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branch2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branch2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branch2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branch2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branch2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branch2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branch2_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branch2_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branch2_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branch2_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branch2_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branch2_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branch2_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_0_modules_branch2_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_0_modules_branch2_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_1_modules_branch2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_1_modules_branch2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_1_modules_branch2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_1_modules_branch2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_1_modules_branch2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_1_modules_branch2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_1_modules_branch2_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_1_modules_branch2_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_1_modules_branch2_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_1_modules_branch2_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_1_modules_branch2_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_1_modules_branch2_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_1_modules_branch2_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_1_modules_branch2_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_1_modules_branch2_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_2_modules_branch2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_2_modules_branch2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_2_modules_branch2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_2_modules_branch2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_2_modules_branch2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_2_modules_branch2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_2_modules_branch2_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_2_modules_branch2_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_2_modules_branch2_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_2_modules_branch2_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_2_modules_branch2_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_2_modules_branch2_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_2_modules_branch2_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_2_modules_branch2_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_2_modules_branch2_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_3_modules_branch2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_3_modules_branch2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_3_modules_branch2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_3_modules_branch2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_3_modules_branch2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_3_modules_branch2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_3_modules_branch2_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_3_modules_branch2_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_3_modules_branch2_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_3_modules_branch2_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_3_modules_branch2_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_3_modules_branch2_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_3_modules_branch2_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_3_modules_branch2_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_3_modules_branch2_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branch1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branch1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branch1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branch1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branch1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branch1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branch1_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branch1_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branch1_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branch1_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branch2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branch2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branch2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branch2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branch2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branch2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branch2_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branch2_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branch2_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branch2_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branch2_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branch2_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branch2_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_0_modules_branch2_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_0_modules_branch2_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branch2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branch2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branch2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branch2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branch2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branch2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branch2_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branch2_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branch2_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branch2_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branch2_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branch2_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branch2_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_1_modules_branch2_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_1_modules_branch2_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branch2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branch2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branch2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branch2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branch2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branch2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branch2_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branch2_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branch2_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branch2_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branch2_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branch2_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branch2_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_2_modules_branch2_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_2_modules_branch2_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_3_modules_branch2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_3_modules_branch2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_3_modules_branch2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_3_modules_branch2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_3_modules_branch2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_3_modules_branch2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_3_modules_branch2_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_3_modules_branch2_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_3_modules_branch2_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_3_modules_branch2_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_3_modules_branch2_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_3_modules_branch2_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_3_modules_branch2_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_3_modules_branch2_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_3_modules_branch2_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_4_modules_branch2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_4_modules_branch2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_4_modules_branch2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_4_modules_branch2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_4_modules_branch2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_4_modules_branch2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_4_modules_branch2_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_4_modules_branch2_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_4_modules_branch2_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_4_modules_branch2_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_4_modules_branch2_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_4_modules_branch2_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_4_modules_branch2_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_4_modules_branch2_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_4_modules_branch2_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_5_modules_branch2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_5_modules_branch2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_5_modules_branch2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_5_modules_branch2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_5_modules_branch2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_5_modules_branch2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_5_modules_branch2_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_5_modules_branch2_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_5_modules_branch2_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_5_modules_branch2_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_5_modules_branch2_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_5_modules_branch2_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_5_modules_branch2_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_5_modules_branch2_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_5_modules_branch2_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_6_modules_branch2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_6_modules_branch2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_6_modules_branch2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_6_modules_branch2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_6_modules_branch2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_6_modules_branch2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_6_modules_branch2_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_6_modules_branch2_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_6_modules_branch2_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_6_modules_branch2_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_6_modules_branch2_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_6_modules_branch2_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_6_modules_branch2_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_6_modules_branch2_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_6_modules_branch2_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_7_modules_branch2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_7_modules_branch2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_7_modules_branch2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_7_modules_branch2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_7_modules_branch2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_7_modules_branch2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_7_modules_branch2_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_7_modules_branch2_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_7_modules_branch2_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_7_modules_branch2_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_7_modules_branch2_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_7_modules_branch2_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_7_modules_branch2_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_7_modules_branch2_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_7_modules_branch2_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branch1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branch1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branch1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branch1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branch1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branch1_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branch1_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branch1_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branch1_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branch1_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branch2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branch2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branch2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branch2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branch2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branch2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branch2_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branch2_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branch2_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branch2_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branch2_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branch2_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branch2_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_0_modules_branch2_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_0_modules_branch2_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branch2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branch2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branch2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branch2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branch2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branch2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branch2_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branch2_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branch2_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branch2_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branch2_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branch2_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branch2_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_1_modules_branch2_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_1_modules_branch2_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_2_modules_branch2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_2_modules_branch2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_2_modules_branch2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_2_modules_branch2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_2_modules_branch2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_2_modules_branch2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_2_modules_branch2_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_2_modules_branch2_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_2_modules_branch2_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_2_modules_branch2_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_2_modules_branch2_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_2_modules_branch2_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_2_modules_branch2_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_2_modules_branch2_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_2_modules_branch2_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_3_modules_branch2_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_3_modules_branch2_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_3_modules_branch2_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_3_modules_branch2_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_3_modules_branch2_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_3_modules_branch2_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_3_modules_branch2_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_3_modules_branch2_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_3_modules_branch2_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_3_modules_branch2_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_3_modules_branch2_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_3_modules_branch2_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_3_modules_branch2_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_3_modules_branch2_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_3_modules_branch2_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv5_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv5_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv5_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_conv5_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv5_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_conv1_modules_0_parameters_weight_ = (
            L_self_modules_conv1_modules_0_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_conv1_modules_1_buffers_running_mean_ = (
            L_self_modules_conv1_modules_1_buffers_running_mean_
        )
        l_self_modules_conv1_modules_1_buffers_running_var_ = (
            L_self_modules_conv1_modules_1_buffers_running_var_
        )
        l_self_modules_conv1_modules_1_parameters_weight_ = (
            L_self_modules_conv1_modules_1_parameters_weight_
        )
        l_self_modules_conv1_modules_1_parameters_bias_ = (
            L_self_modules_conv1_modules_1_parameters_bias_
        )
        l_self_modules_stage2_modules_0_modules_branch1_modules_0_parameters_weight_ = (
            L_self_modules_stage2_modules_0_modules_branch1_modules_0_parameters_weight_
        )
        l_self_modules_stage2_modules_0_modules_branch1_modules_1_buffers_running_mean_ = L_self_modules_stage2_modules_0_modules_branch1_modules_1_buffers_running_mean_
        l_self_modules_stage2_modules_0_modules_branch1_modules_1_buffers_running_var_ = L_self_modules_stage2_modules_0_modules_branch1_modules_1_buffers_running_var_
        l_self_modules_stage2_modules_0_modules_branch1_modules_1_parameters_weight_ = (
            L_self_modules_stage2_modules_0_modules_branch1_modules_1_parameters_weight_
        )
        l_self_modules_stage2_modules_0_modules_branch1_modules_1_parameters_bias_ = (
            L_self_modules_stage2_modules_0_modules_branch1_modules_1_parameters_bias_
        )
        l_self_modules_stage2_modules_0_modules_branch1_modules_2_parameters_weight_ = (
            L_self_modules_stage2_modules_0_modules_branch1_modules_2_parameters_weight_
        )
        l_self_modules_stage2_modules_0_modules_branch1_modules_3_buffers_running_mean_ = L_self_modules_stage2_modules_0_modules_branch1_modules_3_buffers_running_mean_
        l_self_modules_stage2_modules_0_modules_branch1_modules_3_buffers_running_var_ = L_self_modules_stage2_modules_0_modules_branch1_modules_3_buffers_running_var_
        l_self_modules_stage2_modules_0_modules_branch1_modules_3_parameters_weight_ = (
            L_self_modules_stage2_modules_0_modules_branch1_modules_3_parameters_weight_
        )
        l_self_modules_stage2_modules_0_modules_branch1_modules_3_parameters_bias_ = (
            L_self_modules_stage2_modules_0_modules_branch1_modules_3_parameters_bias_
        )
        l_self_modules_stage2_modules_0_modules_branch2_modules_0_parameters_weight_ = (
            L_self_modules_stage2_modules_0_modules_branch2_modules_0_parameters_weight_
        )
        l_self_modules_stage2_modules_0_modules_branch2_modules_1_buffers_running_mean_ = L_self_modules_stage2_modules_0_modules_branch2_modules_1_buffers_running_mean_
        l_self_modules_stage2_modules_0_modules_branch2_modules_1_buffers_running_var_ = L_self_modules_stage2_modules_0_modules_branch2_modules_1_buffers_running_var_
        l_self_modules_stage2_modules_0_modules_branch2_modules_1_parameters_weight_ = (
            L_self_modules_stage2_modules_0_modules_branch2_modules_1_parameters_weight_
        )
        l_self_modules_stage2_modules_0_modules_branch2_modules_1_parameters_bias_ = (
            L_self_modules_stage2_modules_0_modules_branch2_modules_1_parameters_bias_
        )
        l_self_modules_stage2_modules_0_modules_branch2_modules_3_parameters_weight_ = (
            L_self_modules_stage2_modules_0_modules_branch2_modules_3_parameters_weight_
        )
        l_self_modules_stage2_modules_0_modules_branch2_modules_4_buffers_running_mean_ = L_self_modules_stage2_modules_0_modules_branch2_modules_4_buffers_running_mean_
        l_self_modules_stage2_modules_0_modules_branch2_modules_4_buffers_running_var_ = L_self_modules_stage2_modules_0_modules_branch2_modules_4_buffers_running_var_
        l_self_modules_stage2_modules_0_modules_branch2_modules_4_parameters_weight_ = (
            L_self_modules_stage2_modules_0_modules_branch2_modules_4_parameters_weight_
        )
        l_self_modules_stage2_modules_0_modules_branch2_modules_4_parameters_bias_ = (
            L_self_modules_stage2_modules_0_modules_branch2_modules_4_parameters_bias_
        )
        l_self_modules_stage2_modules_0_modules_branch2_modules_5_parameters_weight_ = (
            L_self_modules_stage2_modules_0_modules_branch2_modules_5_parameters_weight_
        )
        l_self_modules_stage2_modules_0_modules_branch2_modules_6_buffers_running_mean_ = L_self_modules_stage2_modules_0_modules_branch2_modules_6_buffers_running_mean_
        l_self_modules_stage2_modules_0_modules_branch2_modules_6_buffers_running_var_ = L_self_modules_stage2_modules_0_modules_branch2_modules_6_buffers_running_var_
        l_self_modules_stage2_modules_0_modules_branch2_modules_6_parameters_weight_ = (
            L_self_modules_stage2_modules_0_modules_branch2_modules_6_parameters_weight_
        )
        l_self_modules_stage2_modules_0_modules_branch2_modules_6_parameters_bias_ = (
            L_self_modules_stage2_modules_0_modules_branch2_modules_6_parameters_bias_
        )
        l_self_modules_stage2_modules_1_modules_branch2_modules_0_parameters_weight_ = (
            L_self_modules_stage2_modules_1_modules_branch2_modules_0_parameters_weight_
        )
        l_self_modules_stage2_modules_1_modules_branch2_modules_1_buffers_running_mean_ = L_self_modules_stage2_modules_1_modules_branch2_modules_1_buffers_running_mean_
        l_self_modules_stage2_modules_1_modules_branch2_modules_1_buffers_running_var_ = L_self_modules_stage2_modules_1_modules_branch2_modules_1_buffers_running_var_
        l_self_modules_stage2_modules_1_modules_branch2_modules_1_parameters_weight_ = (
            L_self_modules_stage2_modules_1_modules_branch2_modules_1_parameters_weight_
        )
        l_self_modules_stage2_modules_1_modules_branch2_modules_1_parameters_bias_ = (
            L_self_modules_stage2_modules_1_modules_branch2_modules_1_parameters_bias_
        )
        l_self_modules_stage2_modules_1_modules_branch2_modules_3_parameters_weight_ = (
            L_self_modules_stage2_modules_1_modules_branch2_modules_3_parameters_weight_
        )
        l_self_modules_stage2_modules_1_modules_branch2_modules_4_buffers_running_mean_ = L_self_modules_stage2_modules_1_modules_branch2_modules_4_buffers_running_mean_
        l_self_modules_stage2_modules_1_modules_branch2_modules_4_buffers_running_var_ = L_self_modules_stage2_modules_1_modules_branch2_modules_4_buffers_running_var_
        l_self_modules_stage2_modules_1_modules_branch2_modules_4_parameters_weight_ = (
            L_self_modules_stage2_modules_1_modules_branch2_modules_4_parameters_weight_
        )
        l_self_modules_stage2_modules_1_modules_branch2_modules_4_parameters_bias_ = (
            L_self_modules_stage2_modules_1_modules_branch2_modules_4_parameters_bias_
        )
        l_self_modules_stage2_modules_1_modules_branch2_modules_5_parameters_weight_ = (
            L_self_modules_stage2_modules_1_modules_branch2_modules_5_parameters_weight_
        )
        l_self_modules_stage2_modules_1_modules_branch2_modules_6_buffers_running_mean_ = L_self_modules_stage2_modules_1_modules_branch2_modules_6_buffers_running_mean_
        l_self_modules_stage2_modules_1_modules_branch2_modules_6_buffers_running_var_ = L_self_modules_stage2_modules_1_modules_branch2_modules_6_buffers_running_var_
        l_self_modules_stage2_modules_1_modules_branch2_modules_6_parameters_weight_ = (
            L_self_modules_stage2_modules_1_modules_branch2_modules_6_parameters_weight_
        )
        l_self_modules_stage2_modules_1_modules_branch2_modules_6_parameters_bias_ = (
            L_self_modules_stage2_modules_1_modules_branch2_modules_6_parameters_bias_
        )
        l_self_modules_stage2_modules_2_modules_branch2_modules_0_parameters_weight_ = (
            L_self_modules_stage2_modules_2_modules_branch2_modules_0_parameters_weight_
        )
        l_self_modules_stage2_modules_2_modules_branch2_modules_1_buffers_running_mean_ = L_self_modules_stage2_modules_2_modules_branch2_modules_1_buffers_running_mean_
        l_self_modules_stage2_modules_2_modules_branch2_modules_1_buffers_running_var_ = L_self_modules_stage2_modules_2_modules_branch2_modules_1_buffers_running_var_
        l_self_modules_stage2_modules_2_modules_branch2_modules_1_parameters_weight_ = (
            L_self_modules_stage2_modules_2_modules_branch2_modules_1_parameters_weight_
        )
        l_self_modules_stage2_modules_2_modules_branch2_modules_1_parameters_bias_ = (
            L_self_modules_stage2_modules_2_modules_branch2_modules_1_parameters_bias_
        )
        l_self_modules_stage2_modules_2_modules_branch2_modules_3_parameters_weight_ = (
            L_self_modules_stage2_modules_2_modules_branch2_modules_3_parameters_weight_
        )
        l_self_modules_stage2_modules_2_modules_branch2_modules_4_buffers_running_mean_ = L_self_modules_stage2_modules_2_modules_branch2_modules_4_buffers_running_mean_
        l_self_modules_stage2_modules_2_modules_branch2_modules_4_buffers_running_var_ = L_self_modules_stage2_modules_2_modules_branch2_modules_4_buffers_running_var_
        l_self_modules_stage2_modules_2_modules_branch2_modules_4_parameters_weight_ = (
            L_self_modules_stage2_modules_2_modules_branch2_modules_4_parameters_weight_
        )
        l_self_modules_stage2_modules_2_modules_branch2_modules_4_parameters_bias_ = (
            L_self_modules_stage2_modules_2_modules_branch2_modules_4_parameters_bias_
        )
        l_self_modules_stage2_modules_2_modules_branch2_modules_5_parameters_weight_ = (
            L_self_modules_stage2_modules_2_modules_branch2_modules_5_parameters_weight_
        )
        l_self_modules_stage2_modules_2_modules_branch2_modules_6_buffers_running_mean_ = L_self_modules_stage2_modules_2_modules_branch2_modules_6_buffers_running_mean_
        l_self_modules_stage2_modules_2_modules_branch2_modules_6_buffers_running_var_ = L_self_modules_stage2_modules_2_modules_branch2_modules_6_buffers_running_var_
        l_self_modules_stage2_modules_2_modules_branch2_modules_6_parameters_weight_ = (
            L_self_modules_stage2_modules_2_modules_branch2_modules_6_parameters_weight_
        )
        l_self_modules_stage2_modules_2_modules_branch2_modules_6_parameters_bias_ = (
            L_self_modules_stage2_modules_2_modules_branch2_modules_6_parameters_bias_
        )
        l_self_modules_stage2_modules_3_modules_branch2_modules_0_parameters_weight_ = (
            L_self_modules_stage2_modules_3_modules_branch2_modules_0_parameters_weight_
        )
        l_self_modules_stage2_modules_3_modules_branch2_modules_1_buffers_running_mean_ = L_self_modules_stage2_modules_3_modules_branch2_modules_1_buffers_running_mean_
        l_self_modules_stage2_modules_3_modules_branch2_modules_1_buffers_running_var_ = L_self_modules_stage2_modules_3_modules_branch2_modules_1_buffers_running_var_
        l_self_modules_stage2_modules_3_modules_branch2_modules_1_parameters_weight_ = (
            L_self_modules_stage2_modules_3_modules_branch2_modules_1_parameters_weight_
        )
        l_self_modules_stage2_modules_3_modules_branch2_modules_1_parameters_bias_ = (
            L_self_modules_stage2_modules_3_modules_branch2_modules_1_parameters_bias_
        )
        l_self_modules_stage2_modules_3_modules_branch2_modules_3_parameters_weight_ = (
            L_self_modules_stage2_modules_3_modules_branch2_modules_3_parameters_weight_
        )
        l_self_modules_stage2_modules_3_modules_branch2_modules_4_buffers_running_mean_ = L_self_modules_stage2_modules_3_modules_branch2_modules_4_buffers_running_mean_
        l_self_modules_stage2_modules_3_modules_branch2_modules_4_buffers_running_var_ = L_self_modules_stage2_modules_3_modules_branch2_modules_4_buffers_running_var_
        l_self_modules_stage2_modules_3_modules_branch2_modules_4_parameters_weight_ = (
            L_self_modules_stage2_modules_3_modules_branch2_modules_4_parameters_weight_
        )
        l_self_modules_stage2_modules_3_modules_branch2_modules_4_parameters_bias_ = (
            L_self_modules_stage2_modules_3_modules_branch2_modules_4_parameters_bias_
        )
        l_self_modules_stage2_modules_3_modules_branch2_modules_5_parameters_weight_ = (
            L_self_modules_stage2_modules_3_modules_branch2_modules_5_parameters_weight_
        )
        l_self_modules_stage2_modules_3_modules_branch2_modules_6_buffers_running_mean_ = L_self_modules_stage2_modules_3_modules_branch2_modules_6_buffers_running_mean_
        l_self_modules_stage2_modules_3_modules_branch2_modules_6_buffers_running_var_ = L_self_modules_stage2_modules_3_modules_branch2_modules_6_buffers_running_var_
        l_self_modules_stage2_modules_3_modules_branch2_modules_6_parameters_weight_ = (
            L_self_modules_stage2_modules_3_modules_branch2_modules_6_parameters_weight_
        )
        l_self_modules_stage2_modules_3_modules_branch2_modules_6_parameters_bias_ = (
            L_self_modules_stage2_modules_3_modules_branch2_modules_6_parameters_bias_
        )
        l_self_modules_stage3_modules_0_modules_branch1_modules_0_parameters_weight_ = (
            L_self_modules_stage3_modules_0_modules_branch1_modules_0_parameters_weight_
        )
        l_self_modules_stage3_modules_0_modules_branch1_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_branch1_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_branch1_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_branch1_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_branch1_modules_1_parameters_weight_ = (
            L_self_modules_stage3_modules_0_modules_branch1_modules_1_parameters_weight_
        )
        l_self_modules_stage3_modules_0_modules_branch1_modules_1_parameters_bias_ = (
            L_self_modules_stage3_modules_0_modules_branch1_modules_1_parameters_bias_
        )
        l_self_modules_stage3_modules_0_modules_branch1_modules_2_parameters_weight_ = (
            L_self_modules_stage3_modules_0_modules_branch1_modules_2_parameters_weight_
        )
        l_self_modules_stage3_modules_0_modules_branch1_modules_3_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_branch1_modules_3_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_branch1_modules_3_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_branch1_modules_3_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_branch1_modules_3_parameters_weight_ = (
            L_self_modules_stage3_modules_0_modules_branch1_modules_3_parameters_weight_
        )
        l_self_modules_stage3_modules_0_modules_branch1_modules_3_parameters_bias_ = (
            L_self_modules_stage3_modules_0_modules_branch1_modules_3_parameters_bias_
        )
        l_self_modules_stage3_modules_0_modules_branch2_modules_0_parameters_weight_ = (
            L_self_modules_stage3_modules_0_modules_branch2_modules_0_parameters_weight_
        )
        l_self_modules_stage3_modules_0_modules_branch2_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_branch2_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_branch2_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_branch2_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_branch2_modules_1_parameters_weight_ = (
            L_self_modules_stage3_modules_0_modules_branch2_modules_1_parameters_weight_
        )
        l_self_modules_stage3_modules_0_modules_branch2_modules_1_parameters_bias_ = (
            L_self_modules_stage3_modules_0_modules_branch2_modules_1_parameters_bias_
        )
        l_self_modules_stage3_modules_0_modules_branch2_modules_3_parameters_weight_ = (
            L_self_modules_stage3_modules_0_modules_branch2_modules_3_parameters_weight_
        )
        l_self_modules_stage3_modules_0_modules_branch2_modules_4_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_branch2_modules_4_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_branch2_modules_4_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_branch2_modules_4_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_branch2_modules_4_parameters_weight_ = (
            L_self_modules_stage3_modules_0_modules_branch2_modules_4_parameters_weight_
        )
        l_self_modules_stage3_modules_0_modules_branch2_modules_4_parameters_bias_ = (
            L_self_modules_stage3_modules_0_modules_branch2_modules_4_parameters_bias_
        )
        l_self_modules_stage3_modules_0_modules_branch2_modules_5_parameters_weight_ = (
            L_self_modules_stage3_modules_0_modules_branch2_modules_5_parameters_weight_
        )
        l_self_modules_stage3_modules_0_modules_branch2_modules_6_buffers_running_mean_ = L_self_modules_stage3_modules_0_modules_branch2_modules_6_buffers_running_mean_
        l_self_modules_stage3_modules_0_modules_branch2_modules_6_buffers_running_var_ = L_self_modules_stage3_modules_0_modules_branch2_modules_6_buffers_running_var_
        l_self_modules_stage3_modules_0_modules_branch2_modules_6_parameters_weight_ = (
            L_self_modules_stage3_modules_0_modules_branch2_modules_6_parameters_weight_
        )
        l_self_modules_stage3_modules_0_modules_branch2_modules_6_parameters_bias_ = (
            L_self_modules_stage3_modules_0_modules_branch2_modules_6_parameters_bias_
        )
        l_self_modules_stage3_modules_1_modules_branch2_modules_0_parameters_weight_ = (
            L_self_modules_stage3_modules_1_modules_branch2_modules_0_parameters_weight_
        )
        l_self_modules_stage3_modules_1_modules_branch2_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_branch2_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_branch2_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_branch2_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_branch2_modules_1_parameters_weight_ = (
            L_self_modules_stage3_modules_1_modules_branch2_modules_1_parameters_weight_
        )
        l_self_modules_stage3_modules_1_modules_branch2_modules_1_parameters_bias_ = (
            L_self_modules_stage3_modules_1_modules_branch2_modules_1_parameters_bias_
        )
        l_self_modules_stage3_modules_1_modules_branch2_modules_3_parameters_weight_ = (
            L_self_modules_stage3_modules_1_modules_branch2_modules_3_parameters_weight_
        )
        l_self_modules_stage3_modules_1_modules_branch2_modules_4_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_branch2_modules_4_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_branch2_modules_4_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_branch2_modules_4_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_branch2_modules_4_parameters_weight_ = (
            L_self_modules_stage3_modules_1_modules_branch2_modules_4_parameters_weight_
        )
        l_self_modules_stage3_modules_1_modules_branch2_modules_4_parameters_bias_ = (
            L_self_modules_stage3_modules_1_modules_branch2_modules_4_parameters_bias_
        )
        l_self_modules_stage3_modules_1_modules_branch2_modules_5_parameters_weight_ = (
            L_self_modules_stage3_modules_1_modules_branch2_modules_5_parameters_weight_
        )
        l_self_modules_stage3_modules_1_modules_branch2_modules_6_buffers_running_mean_ = L_self_modules_stage3_modules_1_modules_branch2_modules_6_buffers_running_mean_
        l_self_modules_stage3_modules_1_modules_branch2_modules_6_buffers_running_var_ = L_self_modules_stage3_modules_1_modules_branch2_modules_6_buffers_running_var_
        l_self_modules_stage3_modules_1_modules_branch2_modules_6_parameters_weight_ = (
            L_self_modules_stage3_modules_1_modules_branch2_modules_6_parameters_weight_
        )
        l_self_modules_stage3_modules_1_modules_branch2_modules_6_parameters_bias_ = (
            L_self_modules_stage3_modules_1_modules_branch2_modules_6_parameters_bias_
        )
        l_self_modules_stage3_modules_2_modules_branch2_modules_0_parameters_weight_ = (
            L_self_modules_stage3_modules_2_modules_branch2_modules_0_parameters_weight_
        )
        l_self_modules_stage3_modules_2_modules_branch2_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_branch2_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_branch2_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_branch2_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_branch2_modules_1_parameters_weight_ = (
            L_self_modules_stage3_modules_2_modules_branch2_modules_1_parameters_weight_
        )
        l_self_modules_stage3_modules_2_modules_branch2_modules_1_parameters_bias_ = (
            L_self_modules_stage3_modules_2_modules_branch2_modules_1_parameters_bias_
        )
        l_self_modules_stage3_modules_2_modules_branch2_modules_3_parameters_weight_ = (
            L_self_modules_stage3_modules_2_modules_branch2_modules_3_parameters_weight_
        )
        l_self_modules_stage3_modules_2_modules_branch2_modules_4_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_branch2_modules_4_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_branch2_modules_4_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_branch2_modules_4_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_branch2_modules_4_parameters_weight_ = (
            L_self_modules_stage3_modules_2_modules_branch2_modules_4_parameters_weight_
        )
        l_self_modules_stage3_modules_2_modules_branch2_modules_4_parameters_bias_ = (
            L_self_modules_stage3_modules_2_modules_branch2_modules_4_parameters_bias_
        )
        l_self_modules_stage3_modules_2_modules_branch2_modules_5_parameters_weight_ = (
            L_self_modules_stage3_modules_2_modules_branch2_modules_5_parameters_weight_
        )
        l_self_modules_stage3_modules_2_modules_branch2_modules_6_buffers_running_mean_ = L_self_modules_stage3_modules_2_modules_branch2_modules_6_buffers_running_mean_
        l_self_modules_stage3_modules_2_modules_branch2_modules_6_buffers_running_var_ = L_self_modules_stage3_modules_2_modules_branch2_modules_6_buffers_running_var_
        l_self_modules_stage3_modules_2_modules_branch2_modules_6_parameters_weight_ = (
            L_self_modules_stage3_modules_2_modules_branch2_modules_6_parameters_weight_
        )
        l_self_modules_stage3_modules_2_modules_branch2_modules_6_parameters_bias_ = (
            L_self_modules_stage3_modules_2_modules_branch2_modules_6_parameters_bias_
        )
        l_self_modules_stage3_modules_3_modules_branch2_modules_0_parameters_weight_ = (
            L_self_modules_stage3_modules_3_modules_branch2_modules_0_parameters_weight_
        )
        l_self_modules_stage3_modules_3_modules_branch2_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_3_modules_branch2_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_3_modules_branch2_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_3_modules_branch2_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_3_modules_branch2_modules_1_parameters_weight_ = (
            L_self_modules_stage3_modules_3_modules_branch2_modules_1_parameters_weight_
        )
        l_self_modules_stage3_modules_3_modules_branch2_modules_1_parameters_bias_ = (
            L_self_modules_stage3_modules_3_modules_branch2_modules_1_parameters_bias_
        )
        l_self_modules_stage3_modules_3_modules_branch2_modules_3_parameters_weight_ = (
            L_self_modules_stage3_modules_3_modules_branch2_modules_3_parameters_weight_
        )
        l_self_modules_stage3_modules_3_modules_branch2_modules_4_buffers_running_mean_ = L_self_modules_stage3_modules_3_modules_branch2_modules_4_buffers_running_mean_
        l_self_modules_stage3_modules_3_modules_branch2_modules_4_buffers_running_var_ = L_self_modules_stage3_modules_3_modules_branch2_modules_4_buffers_running_var_
        l_self_modules_stage3_modules_3_modules_branch2_modules_4_parameters_weight_ = (
            L_self_modules_stage3_modules_3_modules_branch2_modules_4_parameters_weight_
        )
        l_self_modules_stage3_modules_3_modules_branch2_modules_4_parameters_bias_ = (
            L_self_modules_stage3_modules_3_modules_branch2_modules_4_parameters_bias_
        )
        l_self_modules_stage3_modules_3_modules_branch2_modules_5_parameters_weight_ = (
            L_self_modules_stage3_modules_3_modules_branch2_modules_5_parameters_weight_
        )
        l_self_modules_stage3_modules_3_modules_branch2_modules_6_buffers_running_mean_ = L_self_modules_stage3_modules_3_modules_branch2_modules_6_buffers_running_mean_
        l_self_modules_stage3_modules_3_modules_branch2_modules_6_buffers_running_var_ = L_self_modules_stage3_modules_3_modules_branch2_modules_6_buffers_running_var_
        l_self_modules_stage3_modules_3_modules_branch2_modules_6_parameters_weight_ = (
            L_self_modules_stage3_modules_3_modules_branch2_modules_6_parameters_weight_
        )
        l_self_modules_stage3_modules_3_modules_branch2_modules_6_parameters_bias_ = (
            L_self_modules_stage3_modules_3_modules_branch2_modules_6_parameters_bias_
        )
        l_self_modules_stage3_modules_4_modules_branch2_modules_0_parameters_weight_ = (
            L_self_modules_stage3_modules_4_modules_branch2_modules_0_parameters_weight_
        )
        l_self_modules_stage3_modules_4_modules_branch2_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_4_modules_branch2_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_4_modules_branch2_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_4_modules_branch2_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_4_modules_branch2_modules_1_parameters_weight_ = (
            L_self_modules_stage3_modules_4_modules_branch2_modules_1_parameters_weight_
        )
        l_self_modules_stage3_modules_4_modules_branch2_modules_1_parameters_bias_ = (
            L_self_modules_stage3_modules_4_modules_branch2_modules_1_parameters_bias_
        )
        l_self_modules_stage3_modules_4_modules_branch2_modules_3_parameters_weight_ = (
            L_self_modules_stage3_modules_4_modules_branch2_modules_3_parameters_weight_
        )
        l_self_modules_stage3_modules_4_modules_branch2_modules_4_buffers_running_mean_ = L_self_modules_stage3_modules_4_modules_branch2_modules_4_buffers_running_mean_
        l_self_modules_stage3_modules_4_modules_branch2_modules_4_buffers_running_var_ = L_self_modules_stage3_modules_4_modules_branch2_modules_4_buffers_running_var_
        l_self_modules_stage3_modules_4_modules_branch2_modules_4_parameters_weight_ = (
            L_self_modules_stage3_modules_4_modules_branch2_modules_4_parameters_weight_
        )
        l_self_modules_stage3_modules_4_modules_branch2_modules_4_parameters_bias_ = (
            L_self_modules_stage3_modules_4_modules_branch2_modules_4_parameters_bias_
        )
        l_self_modules_stage3_modules_4_modules_branch2_modules_5_parameters_weight_ = (
            L_self_modules_stage3_modules_4_modules_branch2_modules_5_parameters_weight_
        )
        l_self_modules_stage3_modules_4_modules_branch2_modules_6_buffers_running_mean_ = L_self_modules_stage3_modules_4_modules_branch2_modules_6_buffers_running_mean_
        l_self_modules_stage3_modules_4_modules_branch2_modules_6_buffers_running_var_ = L_self_modules_stage3_modules_4_modules_branch2_modules_6_buffers_running_var_
        l_self_modules_stage3_modules_4_modules_branch2_modules_6_parameters_weight_ = (
            L_self_modules_stage3_modules_4_modules_branch2_modules_6_parameters_weight_
        )
        l_self_modules_stage3_modules_4_modules_branch2_modules_6_parameters_bias_ = (
            L_self_modules_stage3_modules_4_modules_branch2_modules_6_parameters_bias_
        )
        l_self_modules_stage3_modules_5_modules_branch2_modules_0_parameters_weight_ = (
            L_self_modules_stage3_modules_5_modules_branch2_modules_0_parameters_weight_
        )
        l_self_modules_stage3_modules_5_modules_branch2_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_5_modules_branch2_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_5_modules_branch2_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_5_modules_branch2_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_5_modules_branch2_modules_1_parameters_weight_ = (
            L_self_modules_stage3_modules_5_modules_branch2_modules_1_parameters_weight_
        )
        l_self_modules_stage3_modules_5_modules_branch2_modules_1_parameters_bias_ = (
            L_self_modules_stage3_modules_5_modules_branch2_modules_1_parameters_bias_
        )
        l_self_modules_stage3_modules_5_modules_branch2_modules_3_parameters_weight_ = (
            L_self_modules_stage3_modules_5_modules_branch2_modules_3_parameters_weight_
        )
        l_self_modules_stage3_modules_5_modules_branch2_modules_4_buffers_running_mean_ = L_self_modules_stage3_modules_5_modules_branch2_modules_4_buffers_running_mean_
        l_self_modules_stage3_modules_5_modules_branch2_modules_4_buffers_running_var_ = L_self_modules_stage3_modules_5_modules_branch2_modules_4_buffers_running_var_
        l_self_modules_stage3_modules_5_modules_branch2_modules_4_parameters_weight_ = (
            L_self_modules_stage3_modules_5_modules_branch2_modules_4_parameters_weight_
        )
        l_self_modules_stage3_modules_5_modules_branch2_modules_4_parameters_bias_ = (
            L_self_modules_stage3_modules_5_modules_branch2_modules_4_parameters_bias_
        )
        l_self_modules_stage3_modules_5_modules_branch2_modules_5_parameters_weight_ = (
            L_self_modules_stage3_modules_5_modules_branch2_modules_5_parameters_weight_
        )
        l_self_modules_stage3_modules_5_modules_branch2_modules_6_buffers_running_mean_ = L_self_modules_stage3_modules_5_modules_branch2_modules_6_buffers_running_mean_
        l_self_modules_stage3_modules_5_modules_branch2_modules_6_buffers_running_var_ = L_self_modules_stage3_modules_5_modules_branch2_modules_6_buffers_running_var_
        l_self_modules_stage3_modules_5_modules_branch2_modules_6_parameters_weight_ = (
            L_self_modules_stage3_modules_5_modules_branch2_modules_6_parameters_weight_
        )
        l_self_modules_stage3_modules_5_modules_branch2_modules_6_parameters_bias_ = (
            L_self_modules_stage3_modules_5_modules_branch2_modules_6_parameters_bias_
        )
        l_self_modules_stage3_modules_6_modules_branch2_modules_0_parameters_weight_ = (
            L_self_modules_stage3_modules_6_modules_branch2_modules_0_parameters_weight_
        )
        l_self_modules_stage3_modules_6_modules_branch2_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_6_modules_branch2_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_6_modules_branch2_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_6_modules_branch2_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_6_modules_branch2_modules_1_parameters_weight_ = (
            L_self_modules_stage3_modules_6_modules_branch2_modules_1_parameters_weight_
        )
        l_self_modules_stage3_modules_6_modules_branch2_modules_1_parameters_bias_ = (
            L_self_modules_stage3_modules_6_modules_branch2_modules_1_parameters_bias_
        )
        l_self_modules_stage3_modules_6_modules_branch2_modules_3_parameters_weight_ = (
            L_self_modules_stage3_modules_6_modules_branch2_modules_3_parameters_weight_
        )
        l_self_modules_stage3_modules_6_modules_branch2_modules_4_buffers_running_mean_ = L_self_modules_stage3_modules_6_modules_branch2_modules_4_buffers_running_mean_
        l_self_modules_stage3_modules_6_modules_branch2_modules_4_buffers_running_var_ = L_self_modules_stage3_modules_6_modules_branch2_modules_4_buffers_running_var_
        l_self_modules_stage3_modules_6_modules_branch2_modules_4_parameters_weight_ = (
            L_self_modules_stage3_modules_6_modules_branch2_modules_4_parameters_weight_
        )
        l_self_modules_stage3_modules_6_modules_branch2_modules_4_parameters_bias_ = (
            L_self_modules_stage3_modules_6_modules_branch2_modules_4_parameters_bias_
        )
        l_self_modules_stage3_modules_6_modules_branch2_modules_5_parameters_weight_ = (
            L_self_modules_stage3_modules_6_modules_branch2_modules_5_parameters_weight_
        )
        l_self_modules_stage3_modules_6_modules_branch2_modules_6_buffers_running_mean_ = L_self_modules_stage3_modules_6_modules_branch2_modules_6_buffers_running_mean_
        l_self_modules_stage3_modules_6_modules_branch2_modules_6_buffers_running_var_ = L_self_modules_stage3_modules_6_modules_branch2_modules_6_buffers_running_var_
        l_self_modules_stage3_modules_6_modules_branch2_modules_6_parameters_weight_ = (
            L_self_modules_stage3_modules_6_modules_branch2_modules_6_parameters_weight_
        )
        l_self_modules_stage3_modules_6_modules_branch2_modules_6_parameters_bias_ = (
            L_self_modules_stage3_modules_6_modules_branch2_modules_6_parameters_bias_
        )
        l_self_modules_stage3_modules_7_modules_branch2_modules_0_parameters_weight_ = (
            L_self_modules_stage3_modules_7_modules_branch2_modules_0_parameters_weight_
        )
        l_self_modules_stage3_modules_7_modules_branch2_modules_1_buffers_running_mean_ = L_self_modules_stage3_modules_7_modules_branch2_modules_1_buffers_running_mean_
        l_self_modules_stage3_modules_7_modules_branch2_modules_1_buffers_running_var_ = L_self_modules_stage3_modules_7_modules_branch2_modules_1_buffers_running_var_
        l_self_modules_stage3_modules_7_modules_branch2_modules_1_parameters_weight_ = (
            L_self_modules_stage3_modules_7_modules_branch2_modules_1_parameters_weight_
        )
        l_self_modules_stage3_modules_7_modules_branch2_modules_1_parameters_bias_ = (
            L_self_modules_stage3_modules_7_modules_branch2_modules_1_parameters_bias_
        )
        l_self_modules_stage3_modules_7_modules_branch2_modules_3_parameters_weight_ = (
            L_self_modules_stage3_modules_7_modules_branch2_modules_3_parameters_weight_
        )
        l_self_modules_stage3_modules_7_modules_branch2_modules_4_buffers_running_mean_ = L_self_modules_stage3_modules_7_modules_branch2_modules_4_buffers_running_mean_
        l_self_modules_stage3_modules_7_modules_branch2_modules_4_buffers_running_var_ = L_self_modules_stage3_modules_7_modules_branch2_modules_4_buffers_running_var_
        l_self_modules_stage3_modules_7_modules_branch2_modules_4_parameters_weight_ = (
            L_self_modules_stage3_modules_7_modules_branch2_modules_4_parameters_weight_
        )
        l_self_modules_stage3_modules_7_modules_branch2_modules_4_parameters_bias_ = (
            L_self_modules_stage3_modules_7_modules_branch2_modules_4_parameters_bias_
        )
        l_self_modules_stage3_modules_7_modules_branch2_modules_5_parameters_weight_ = (
            L_self_modules_stage3_modules_7_modules_branch2_modules_5_parameters_weight_
        )
        l_self_modules_stage3_modules_7_modules_branch2_modules_6_buffers_running_mean_ = L_self_modules_stage3_modules_7_modules_branch2_modules_6_buffers_running_mean_
        l_self_modules_stage3_modules_7_modules_branch2_modules_6_buffers_running_var_ = L_self_modules_stage3_modules_7_modules_branch2_modules_6_buffers_running_var_
        l_self_modules_stage3_modules_7_modules_branch2_modules_6_parameters_weight_ = (
            L_self_modules_stage3_modules_7_modules_branch2_modules_6_parameters_weight_
        )
        l_self_modules_stage3_modules_7_modules_branch2_modules_6_parameters_bias_ = (
            L_self_modules_stage3_modules_7_modules_branch2_modules_6_parameters_bias_
        )
        l_self_modules_stage4_modules_0_modules_branch1_modules_0_parameters_weight_ = (
            L_self_modules_stage4_modules_0_modules_branch1_modules_0_parameters_weight_
        )
        l_self_modules_stage4_modules_0_modules_branch1_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branch1_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branch1_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branch1_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branch1_modules_1_parameters_weight_ = (
            L_self_modules_stage4_modules_0_modules_branch1_modules_1_parameters_weight_
        )
        l_self_modules_stage4_modules_0_modules_branch1_modules_1_parameters_bias_ = (
            L_self_modules_stage4_modules_0_modules_branch1_modules_1_parameters_bias_
        )
        l_self_modules_stage4_modules_0_modules_branch1_modules_2_parameters_weight_ = (
            L_self_modules_stage4_modules_0_modules_branch1_modules_2_parameters_weight_
        )
        l_self_modules_stage4_modules_0_modules_branch1_modules_3_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branch1_modules_3_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branch1_modules_3_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branch1_modules_3_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branch1_modules_3_parameters_weight_ = (
            L_self_modules_stage4_modules_0_modules_branch1_modules_3_parameters_weight_
        )
        l_self_modules_stage4_modules_0_modules_branch1_modules_3_parameters_bias_ = (
            L_self_modules_stage4_modules_0_modules_branch1_modules_3_parameters_bias_
        )
        l_self_modules_stage4_modules_0_modules_branch2_modules_0_parameters_weight_ = (
            L_self_modules_stage4_modules_0_modules_branch2_modules_0_parameters_weight_
        )
        l_self_modules_stage4_modules_0_modules_branch2_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branch2_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branch2_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branch2_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branch2_modules_1_parameters_weight_ = (
            L_self_modules_stage4_modules_0_modules_branch2_modules_1_parameters_weight_
        )
        l_self_modules_stage4_modules_0_modules_branch2_modules_1_parameters_bias_ = (
            L_self_modules_stage4_modules_0_modules_branch2_modules_1_parameters_bias_
        )
        l_self_modules_stage4_modules_0_modules_branch2_modules_3_parameters_weight_ = (
            L_self_modules_stage4_modules_0_modules_branch2_modules_3_parameters_weight_
        )
        l_self_modules_stage4_modules_0_modules_branch2_modules_4_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branch2_modules_4_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branch2_modules_4_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branch2_modules_4_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branch2_modules_4_parameters_weight_ = (
            L_self_modules_stage4_modules_0_modules_branch2_modules_4_parameters_weight_
        )
        l_self_modules_stage4_modules_0_modules_branch2_modules_4_parameters_bias_ = (
            L_self_modules_stage4_modules_0_modules_branch2_modules_4_parameters_bias_
        )
        l_self_modules_stage4_modules_0_modules_branch2_modules_5_parameters_weight_ = (
            L_self_modules_stage4_modules_0_modules_branch2_modules_5_parameters_weight_
        )
        l_self_modules_stage4_modules_0_modules_branch2_modules_6_buffers_running_mean_ = L_self_modules_stage4_modules_0_modules_branch2_modules_6_buffers_running_mean_
        l_self_modules_stage4_modules_0_modules_branch2_modules_6_buffers_running_var_ = L_self_modules_stage4_modules_0_modules_branch2_modules_6_buffers_running_var_
        l_self_modules_stage4_modules_0_modules_branch2_modules_6_parameters_weight_ = (
            L_self_modules_stage4_modules_0_modules_branch2_modules_6_parameters_weight_
        )
        l_self_modules_stage4_modules_0_modules_branch2_modules_6_parameters_bias_ = (
            L_self_modules_stage4_modules_0_modules_branch2_modules_6_parameters_bias_
        )
        l_self_modules_stage4_modules_1_modules_branch2_modules_0_parameters_weight_ = (
            L_self_modules_stage4_modules_1_modules_branch2_modules_0_parameters_weight_
        )
        l_self_modules_stage4_modules_1_modules_branch2_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branch2_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branch2_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branch2_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branch2_modules_1_parameters_weight_ = (
            L_self_modules_stage4_modules_1_modules_branch2_modules_1_parameters_weight_
        )
        l_self_modules_stage4_modules_1_modules_branch2_modules_1_parameters_bias_ = (
            L_self_modules_stage4_modules_1_modules_branch2_modules_1_parameters_bias_
        )
        l_self_modules_stage4_modules_1_modules_branch2_modules_3_parameters_weight_ = (
            L_self_modules_stage4_modules_1_modules_branch2_modules_3_parameters_weight_
        )
        l_self_modules_stage4_modules_1_modules_branch2_modules_4_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branch2_modules_4_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branch2_modules_4_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branch2_modules_4_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branch2_modules_4_parameters_weight_ = (
            L_self_modules_stage4_modules_1_modules_branch2_modules_4_parameters_weight_
        )
        l_self_modules_stage4_modules_1_modules_branch2_modules_4_parameters_bias_ = (
            L_self_modules_stage4_modules_1_modules_branch2_modules_4_parameters_bias_
        )
        l_self_modules_stage4_modules_1_modules_branch2_modules_5_parameters_weight_ = (
            L_self_modules_stage4_modules_1_modules_branch2_modules_5_parameters_weight_
        )
        l_self_modules_stage4_modules_1_modules_branch2_modules_6_buffers_running_mean_ = L_self_modules_stage4_modules_1_modules_branch2_modules_6_buffers_running_mean_
        l_self_modules_stage4_modules_1_modules_branch2_modules_6_buffers_running_var_ = L_self_modules_stage4_modules_1_modules_branch2_modules_6_buffers_running_var_
        l_self_modules_stage4_modules_1_modules_branch2_modules_6_parameters_weight_ = (
            L_self_modules_stage4_modules_1_modules_branch2_modules_6_parameters_weight_
        )
        l_self_modules_stage4_modules_1_modules_branch2_modules_6_parameters_bias_ = (
            L_self_modules_stage4_modules_1_modules_branch2_modules_6_parameters_bias_
        )
        l_self_modules_stage4_modules_2_modules_branch2_modules_0_parameters_weight_ = (
            L_self_modules_stage4_modules_2_modules_branch2_modules_0_parameters_weight_
        )
        l_self_modules_stage4_modules_2_modules_branch2_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_2_modules_branch2_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_2_modules_branch2_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_2_modules_branch2_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_2_modules_branch2_modules_1_parameters_weight_ = (
            L_self_modules_stage4_modules_2_modules_branch2_modules_1_parameters_weight_
        )
        l_self_modules_stage4_modules_2_modules_branch2_modules_1_parameters_bias_ = (
            L_self_modules_stage4_modules_2_modules_branch2_modules_1_parameters_bias_
        )
        l_self_modules_stage4_modules_2_modules_branch2_modules_3_parameters_weight_ = (
            L_self_modules_stage4_modules_2_modules_branch2_modules_3_parameters_weight_
        )
        l_self_modules_stage4_modules_2_modules_branch2_modules_4_buffers_running_mean_ = L_self_modules_stage4_modules_2_modules_branch2_modules_4_buffers_running_mean_
        l_self_modules_stage4_modules_2_modules_branch2_modules_4_buffers_running_var_ = L_self_modules_stage4_modules_2_modules_branch2_modules_4_buffers_running_var_
        l_self_modules_stage4_modules_2_modules_branch2_modules_4_parameters_weight_ = (
            L_self_modules_stage4_modules_2_modules_branch2_modules_4_parameters_weight_
        )
        l_self_modules_stage4_modules_2_modules_branch2_modules_4_parameters_bias_ = (
            L_self_modules_stage4_modules_2_modules_branch2_modules_4_parameters_bias_
        )
        l_self_modules_stage4_modules_2_modules_branch2_modules_5_parameters_weight_ = (
            L_self_modules_stage4_modules_2_modules_branch2_modules_5_parameters_weight_
        )
        l_self_modules_stage4_modules_2_modules_branch2_modules_6_buffers_running_mean_ = L_self_modules_stage4_modules_2_modules_branch2_modules_6_buffers_running_mean_
        l_self_modules_stage4_modules_2_modules_branch2_modules_6_buffers_running_var_ = L_self_modules_stage4_modules_2_modules_branch2_modules_6_buffers_running_var_
        l_self_modules_stage4_modules_2_modules_branch2_modules_6_parameters_weight_ = (
            L_self_modules_stage4_modules_2_modules_branch2_modules_6_parameters_weight_
        )
        l_self_modules_stage4_modules_2_modules_branch2_modules_6_parameters_bias_ = (
            L_self_modules_stage4_modules_2_modules_branch2_modules_6_parameters_bias_
        )
        l_self_modules_stage4_modules_3_modules_branch2_modules_0_parameters_weight_ = (
            L_self_modules_stage4_modules_3_modules_branch2_modules_0_parameters_weight_
        )
        l_self_modules_stage4_modules_3_modules_branch2_modules_1_buffers_running_mean_ = L_self_modules_stage4_modules_3_modules_branch2_modules_1_buffers_running_mean_
        l_self_modules_stage4_modules_3_modules_branch2_modules_1_buffers_running_var_ = L_self_modules_stage4_modules_3_modules_branch2_modules_1_buffers_running_var_
        l_self_modules_stage4_modules_3_modules_branch2_modules_1_parameters_weight_ = (
            L_self_modules_stage4_modules_3_modules_branch2_modules_1_parameters_weight_
        )
        l_self_modules_stage4_modules_3_modules_branch2_modules_1_parameters_bias_ = (
            L_self_modules_stage4_modules_3_modules_branch2_modules_1_parameters_bias_
        )
        l_self_modules_stage4_modules_3_modules_branch2_modules_3_parameters_weight_ = (
            L_self_modules_stage4_modules_3_modules_branch2_modules_3_parameters_weight_
        )
        l_self_modules_stage4_modules_3_modules_branch2_modules_4_buffers_running_mean_ = L_self_modules_stage4_modules_3_modules_branch2_modules_4_buffers_running_mean_
        l_self_modules_stage4_modules_3_modules_branch2_modules_4_buffers_running_var_ = L_self_modules_stage4_modules_3_modules_branch2_modules_4_buffers_running_var_
        l_self_modules_stage4_modules_3_modules_branch2_modules_4_parameters_weight_ = (
            L_self_modules_stage4_modules_3_modules_branch2_modules_4_parameters_weight_
        )
        l_self_modules_stage4_modules_3_modules_branch2_modules_4_parameters_bias_ = (
            L_self_modules_stage4_modules_3_modules_branch2_modules_4_parameters_bias_
        )
        l_self_modules_stage4_modules_3_modules_branch2_modules_5_parameters_weight_ = (
            L_self_modules_stage4_modules_3_modules_branch2_modules_5_parameters_weight_
        )
        l_self_modules_stage4_modules_3_modules_branch2_modules_6_buffers_running_mean_ = L_self_modules_stage4_modules_3_modules_branch2_modules_6_buffers_running_mean_
        l_self_modules_stage4_modules_3_modules_branch2_modules_6_buffers_running_var_ = L_self_modules_stage4_modules_3_modules_branch2_modules_6_buffers_running_var_
        l_self_modules_stage4_modules_3_modules_branch2_modules_6_parameters_weight_ = (
            L_self_modules_stage4_modules_3_modules_branch2_modules_6_parameters_weight_
        )
        l_self_modules_stage4_modules_3_modules_branch2_modules_6_parameters_bias_ = (
            L_self_modules_stage4_modules_3_modules_branch2_modules_6_parameters_bias_
        )
        l_self_modules_conv5_modules_0_parameters_weight_ = (
            L_self_modules_conv5_modules_0_parameters_weight_
        )
        l_self_modules_conv5_modules_1_buffers_running_mean_ = (
            L_self_modules_conv5_modules_1_buffers_running_mean_
        )
        l_self_modules_conv5_modules_1_buffers_running_var_ = (
            L_self_modules_conv5_modules_1_buffers_running_var_
        )
        l_self_modules_conv5_modules_1_parameters_weight_ = (
            L_self_modules_conv5_modules_1_parameters_weight_
        )
        l_self_modules_conv5_modules_1_parameters_bias_ = (
            L_self_modules_conv5_modules_1_parameters_bias_
        )
        l_self_modules_fc_parameters_weight_ = L_self_modules_fc_parameters_weight_
        l_self_modules_fc_parameters_bias_ = L_self_modules_fc_parameters_bias_
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_conv1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_conv1_modules_0_parameters_weight_ = None
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_conv1_modules_1_buffers_running_mean_,
            l_self_modules_conv1_modules_1_buffers_running_var_,
            l_self_modules_conv1_modules_1_parameters_weight_,
            l_self_modules_conv1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = (
            l_self_modules_conv1_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_conv1_modules_1_buffers_running_var_
        ) = (
            l_self_modules_conv1_modules_1_parameters_weight_
        ) = l_self_modules_conv1_modules_1_parameters_bias_ = None
        input_3 = torch.nn.functional.relu(input_2, inplace=True)
        input_2 = None
        x = torch.nn.functional.max_pool2d(
            input_3, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        input_3 = None
        input_4 = torch.conv2d(
            x,
            l_self_modules_stage2_modules_0_modules_branch1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            24,
        )
        l_self_modules_stage2_modules_0_modules_branch1_modules_0_parameters_weight_ = (
            None
        )
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_stage2_modules_0_modules_branch1_modules_1_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branch1_modules_1_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branch1_modules_1_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branch1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_4 = l_self_modules_stage2_modules_0_modules_branch1_modules_1_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branch1_modules_1_buffers_running_var_ = (
            l_self_modules_stage2_modules_0_modules_branch1_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_0_modules_branch1_modules_1_parameters_bias_
        ) = None
        input_6 = torch.conv2d(
            input_5,
            l_self_modules_stage2_modules_0_modules_branch1_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_5 = (
            l_self_modules_stage2_modules_0_modules_branch1_modules_2_parameters_weight_
        ) = None
        input_7 = torch.nn.functional.batch_norm(
            input_6,
            l_self_modules_stage2_modules_0_modules_branch1_modules_3_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branch1_modules_3_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branch1_modules_3_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branch1_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_6 = l_self_modules_stage2_modules_0_modules_branch1_modules_3_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branch1_modules_3_buffers_running_var_ = (
            l_self_modules_stage2_modules_0_modules_branch1_modules_3_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_0_modules_branch1_modules_3_parameters_bias_
        ) = None
        input_8 = torch.nn.functional.relu(input_7, inplace=True)
        input_7 = None
        input_9 = torch.conv2d(
            x,
            l_self_modules_stage2_modules_0_modules_branch2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x = (
            l_self_modules_stage2_modules_0_modules_branch2_modules_0_parameters_weight_
        ) = None
        input_10 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_stage2_modules_0_modules_branch2_modules_1_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branch2_modules_1_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branch2_modules_1_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branch2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_9 = l_self_modules_stage2_modules_0_modules_branch2_modules_1_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branch2_modules_1_buffers_running_var_ = (
            l_self_modules_stage2_modules_0_modules_branch2_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_0_modules_branch2_modules_1_parameters_bias_
        ) = None
        input_11 = torch.nn.functional.relu(input_10, inplace=True)
        input_10 = None
        input_12 = torch.conv2d(
            input_11,
            l_self_modules_stage2_modules_0_modules_branch2_modules_3_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            58,
        )
        input_11 = (
            l_self_modules_stage2_modules_0_modules_branch2_modules_3_parameters_weight_
        ) = None
        input_13 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_stage2_modules_0_modules_branch2_modules_4_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branch2_modules_4_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branch2_modules_4_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branch2_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_12 = l_self_modules_stage2_modules_0_modules_branch2_modules_4_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branch2_modules_4_buffers_running_var_ = (
            l_self_modules_stage2_modules_0_modules_branch2_modules_4_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_0_modules_branch2_modules_4_parameters_bias_
        ) = None
        input_14 = torch.conv2d(
            input_13,
            l_self_modules_stage2_modules_0_modules_branch2_modules_5_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_13 = (
            l_self_modules_stage2_modules_0_modules_branch2_modules_5_parameters_weight_
        ) = None
        input_15 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_stage2_modules_0_modules_branch2_modules_6_buffers_running_mean_,
            l_self_modules_stage2_modules_0_modules_branch2_modules_6_buffers_running_var_,
            l_self_modules_stage2_modules_0_modules_branch2_modules_6_parameters_weight_,
            l_self_modules_stage2_modules_0_modules_branch2_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_14 = l_self_modules_stage2_modules_0_modules_branch2_modules_6_buffers_running_mean_ = l_self_modules_stage2_modules_0_modules_branch2_modules_6_buffers_running_var_ = (
            l_self_modules_stage2_modules_0_modules_branch2_modules_6_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_0_modules_branch2_modules_6_parameters_bias_
        ) = None
        input_16 = torch.nn.functional.relu(input_15, inplace=True)
        input_15 = None
        out = torch.cat((input_8, input_16), dim=1)
        input_8 = input_16 = None
        x_1 = out.view(1, 2, 58, 28, 28)
        out = None
        transpose = torch.transpose(x_1, 1, 2)
        x_1 = None
        x_2 = transpose.contiguous()
        transpose = None
        x_3 = x_2.view(1, 116, 28, 28)
        x_2 = None
        chunk = x_3.chunk(2, dim=1)
        x_3 = None
        x1 = chunk[0]
        x2 = chunk[1]
        chunk = None
        input_17 = torch.conv2d(
            x2,
            l_self_modules_stage2_modules_1_modules_branch2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x2 = (
            l_self_modules_stage2_modules_1_modules_branch2_modules_0_parameters_weight_
        ) = None
        input_18 = torch.nn.functional.batch_norm(
            input_17,
            l_self_modules_stage2_modules_1_modules_branch2_modules_1_buffers_running_mean_,
            l_self_modules_stage2_modules_1_modules_branch2_modules_1_buffers_running_var_,
            l_self_modules_stage2_modules_1_modules_branch2_modules_1_parameters_weight_,
            l_self_modules_stage2_modules_1_modules_branch2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_17 = l_self_modules_stage2_modules_1_modules_branch2_modules_1_buffers_running_mean_ = l_self_modules_stage2_modules_1_modules_branch2_modules_1_buffers_running_var_ = (
            l_self_modules_stage2_modules_1_modules_branch2_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_1_modules_branch2_modules_1_parameters_bias_
        ) = None
        input_19 = torch.nn.functional.relu(input_18, inplace=True)
        input_18 = None
        input_20 = torch.conv2d(
            input_19,
            l_self_modules_stage2_modules_1_modules_branch2_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            58,
        )
        input_19 = (
            l_self_modules_stage2_modules_1_modules_branch2_modules_3_parameters_weight_
        ) = None
        input_21 = torch.nn.functional.batch_norm(
            input_20,
            l_self_modules_stage2_modules_1_modules_branch2_modules_4_buffers_running_mean_,
            l_self_modules_stage2_modules_1_modules_branch2_modules_4_buffers_running_var_,
            l_self_modules_stage2_modules_1_modules_branch2_modules_4_parameters_weight_,
            l_self_modules_stage2_modules_1_modules_branch2_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_20 = l_self_modules_stage2_modules_1_modules_branch2_modules_4_buffers_running_mean_ = l_self_modules_stage2_modules_1_modules_branch2_modules_4_buffers_running_var_ = (
            l_self_modules_stage2_modules_1_modules_branch2_modules_4_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_1_modules_branch2_modules_4_parameters_bias_
        ) = None
        input_22 = torch.conv2d(
            input_21,
            l_self_modules_stage2_modules_1_modules_branch2_modules_5_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_21 = (
            l_self_modules_stage2_modules_1_modules_branch2_modules_5_parameters_weight_
        ) = None
        input_23 = torch.nn.functional.batch_norm(
            input_22,
            l_self_modules_stage2_modules_1_modules_branch2_modules_6_buffers_running_mean_,
            l_self_modules_stage2_modules_1_modules_branch2_modules_6_buffers_running_var_,
            l_self_modules_stage2_modules_1_modules_branch2_modules_6_parameters_weight_,
            l_self_modules_stage2_modules_1_modules_branch2_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_22 = l_self_modules_stage2_modules_1_modules_branch2_modules_6_buffers_running_mean_ = l_self_modules_stage2_modules_1_modules_branch2_modules_6_buffers_running_var_ = (
            l_self_modules_stage2_modules_1_modules_branch2_modules_6_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_1_modules_branch2_modules_6_parameters_bias_
        ) = None
        input_24 = torch.nn.functional.relu(input_23, inplace=True)
        input_23 = None
        out_1 = torch.cat((x1, input_24), dim=1)
        x1 = input_24 = None
        x_4 = out_1.view(1, 2, 58, 28, 28)
        out_1 = None
        transpose_1 = torch.transpose(x_4, 1, 2)
        x_4 = None
        x_5 = transpose_1.contiguous()
        transpose_1 = None
        x_6 = x_5.view(1, 116, 28, 28)
        x_5 = None
        chunk_1 = x_6.chunk(2, dim=1)
        x_6 = None
        x1_1 = chunk_1[0]
        x2_1 = chunk_1[1]
        chunk_1 = None
        input_25 = torch.conv2d(
            x2_1,
            l_self_modules_stage2_modules_2_modules_branch2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x2_1 = (
            l_self_modules_stage2_modules_2_modules_branch2_modules_0_parameters_weight_
        ) = None
        input_26 = torch.nn.functional.batch_norm(
            input_25,
            l_self_modules_stage2_modules_2_modules_branch2_modules_1_buffers_running_mean_,
            l_self_modules_stage2_modules_2_modules_branch2_modules_1_buffers_running_var_,
            l_self_modules_stage2_modules_2_modules_branch2_modules_1_parameters_weight_,
            l_self_modules_stage2_modules_2_modules_branch2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_25 = l_self_modules_stage2_modules_2_modules_branch2_modules_1_buffers_running_mean_ = l_self_modules_stage2_modules_2_modules_branch2_modules_1_buffers_running_var_ = (
            l_self_modules_stage2_modules_2_modules_branch2_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_2_modules_branch2_modules_1_parameters_bias_
        ) = None
        input_27 = torch.nn.functional.relu(input_26, inplace=True)
        input_26 = None
        input_28 = torch.conv2d(
            input_27,
            l_self_modules_stage2_modules_2_modules_branch2_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            58,
        )
        input_27 = (
            l_self_modules_stage2_modules_2_modules_branch2_modules_3_parameters_weight_
        ) = None
        input_29 = torch.nn.functional.batch_norm(
            input_28,
            l_self_modules_stage2_modules_2_modules_branch2_modules_4_buffers_running_mean_,
            l_self_modules_stage2_modules_2_modules_branch2_modules_4_buffers_running_var_,
            l_self_modules_stage2_modules_2_modules_branch2_modules_4_parameters_weight_,
            l_self_modules_stage2_modules_2_modules_branch2_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_28 = l_self_modules_stage2_modules_2_modules_branch2_modules_4_buffers_running_mean_ = l_self_modules_stage2_modules_2_modules_branch2_modules_4_buffers_running_var_ = (
            l_self_modules_stage2_modules_2_modules_branch2_modules_4_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_2_modules_branch2_modules_4_parameters_bias_
        ) = None
        input_30 = torch.conv2d(
            input_29,
            l_self_modules_stage2_modules_2_modules_branch2_modules_5_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_29 = (
            l_self_modules_stage2_modules_2_modules_branch2_modules_5_parameters_weight_
        ) = None
        input_31 = torch.nn.functional.batch_norm(
            input_30,
            l_self_modules_stage2_modules_2_modules_branch2_modules_6_buffers_running_mean_,
            l_self_modules_stage2_modules_2_modules_branch2_modules_6_buffers_running_var_,
            l_self_modules_stage2_modules_2_modules_branch2_modules_6_parameters_weight_,
            l_self_modules_stage2_modules_2_modules_branch2_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_30 = l_self_modules_stage2_modules_2_modules_branch2_modules_6_buffers_running_mean_ = l_self_modules_stage2_modules_2_modules_branch2_modules_6_buffers_running_var_ = (
            l_self_modules_stage2_modules_2_modules_branch2_modules_6_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_2_modules_branch2_modules_6_parameters_bias_
        ) = None
        input_32 = torch.nn.functional.relu(input_31, inplace=True)
        input_31 = None
        out_2 = torch.cat((x1_1, input_32), dim=1)
        x1_1 = input_32 = None
        x_7 = out_2.view(1, 2, 58, 28, 28)
        out_2 = None
        transpose_2 = torch.transpose(x_7, 1, 2)
        x_7 = None
        x_8 = transpose_2.contiguous()
        transpose_2 = None
        x_9 = x_8.view(1, 116, 28, 28)
        x_8 = None
        chunk_2 = x_9.chunk(2, dim=1)
        x_9 = None
        x1_2 = chunk_2[0]
        x2_2 = chunk_2[1]
        chunk_2 = None
        input_33 = torch.conv2d(
            x2_2,
            l_self_modules_stage2_modules_3_modules_branch2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x2_2 = (
            l_self_modules_stage2_modules_3_modules_branch2_modules_0_parameters_weight_
        ) = None
        input_34 = torch.nn.functional.batch_norm(
            input_33,
            l_self_modules_stage2_modules_3_modules_branch2_modules_1_buffers_running_mean_,
            l_self_modules_stage2_modules_3_modules_branch2_modules_1_buffers_running_var_,
            l_self_modules_stage2_modules_3_modules_branch2_modules_1_parameters_weight_,
            l_self_modules_stage2_modules_3_modules_branch2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_33 = l_self_modules_stage2_modules_3_modules_branch2_modules_1_buffers_running_mean_ = l_self_modules_stage2_modules_3_modules_branch2_modules_1_buffers_running_var_ = (
            l_self_modules_stage2_modules_3_modules_branch2_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_3_modules_branch2_modules_1_parameters_bias_
        ) = None
        input_35 = torch.nn.functional.relu(input_34, inplace=True)
        input_34 = None
        input_36 = torch.conv2d(
            input_35,
            l_self_modules_stage2_modules_3_modules_branch2_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            58,
        )
        input_35 = (
            l_self_modules_stage2_modules_3_modules_branch2_modules_3_parameters_weight_
        ) = None
        input_37 = torch.nn.functional.batch_norm(
            input_36,
            l_self_modules_stage2_modules_3_modules_branch2_modules_4_buffers_running_mean_,
            l_self_modules_stage2_modules_3_modules_branch2_modules_4_buffers_running_var_,
            l_self_modules_stage2_modules_3_modules_branch2_modules_4_parameters_weight_,
            l_self_modules_stage2_modules_3_modules_branch2_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_36 = l_self_modules_stage2_modules_3_modules_branch2_modules_4_buffers_running_mean_ = l_self_modules_stage2_modules_3_modules_branch2_modules_4_buffers_running_var_ = (
            l_self_modules_stage2_modules_3_modules_branch2_modules_4_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_3_modules_branch2_modules_4_parameters_bias_
        ) = None
        input_38 = torch.conv2d(
            input_37,
            l_self_modules_stage2_modules_3_modules_branch2_modules_5_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_37 = (
            l_self_modules_stage2_modules_3_modules_branch2_modules_5_parameters_weight_
        ) = None
        input_39 = torch.nn.functional.batch_norm(
            input_38,
            l_self_modules_stage2_modules_3_modules_branch2_modules_6_buffers_running_mean_,
            l_self_modules_stage2_modules_3_modules_branch2_modules_6_buffers_running_var_,
            l_self_modules_stage2_modules_3_modules_branch2_modules_6_parameters_weight_,
            l_self_modules_stage2_modules_3_modules_branch2_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_38 = l_self_modules_stage2_modules_3_modules_branch2_modules_6_buffers_running_mean_ = l_self_modules_stage2_modules_3_modules_branch2_modules_6_buffers_running_var_ = (
            l_self_modules_stage2_modules_3_modules_branch2_modules_6_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_3_modules_branch2_modules_6_parameters_bias_
        ) = None
        input_40 = torch.nn.functional.relu(input_39, inplace=True)
        input_39 = None
        out_3 = torch.cat((x1_2, input_40), dim=1)
        x1_2 = input_40 = None
        x_10 = out_3.view(1, 2, 58, 28, 28)
        out_3 = None
        transpose_3 = torch.transpose(x_10, 1, 2)
        x_10 = None
        x_11 = transpose_3.contiguous()
        transpose_3 = None
        x_12 = x_11.view(1, 116, 28, 28)
        x_11 = None
        input_41 = torch.conv2d(
            x_12,
            l_self_modules_stage3_modules_0_modules_branch1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            116,
        )
        l_self_modules_stage3_modules_0_modules_branch1_modules_0_parameters_weight_ = (
            None
        )
        input_42 = torch.nn.functional.batch_norm(
            input_41,
            l_self_modules_stage3_modules_0_modules_branch1_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branch1_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branch1_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branch1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_41 = l_self_modules_stage3_modules_0_modules_branch1_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branch1_modules_1_buffers_running_var_ = (
            l_self_modules_stage3_modules_0_modules_branch1_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_0_modules_branch1_modules_1_parameters_bias_
        ) = None
        input_43 = torch.conv2d(
            input_42,
            l_self_modules_stage3_modules_0_modules_branch1_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_42 = (
            l_self_modules_stage3_modules_0_modules_branch1_modules_2_parameters_weight_
        ) = None
        input_44 = torch.nn.functional.batch_norm(
            input_43,
            l_self_modules_stage3_modules_0_modules_branch1_modules_3_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branch1_modules_3_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branch1_modules_3_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branch1_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_43 = l_self_modules_stage3_modules_0_modules_branch1_modules_3_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branch1_modules_3_buffers_running_var_ = (
            l_self_modules_stage3_modules_0_modules_branch1_modules_3_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_0_modules_branch1_modules_3_parameters_bias_
        ) = None
        input_45 = torch.nn.functional.relu(input_44, inplace=True)
        input_44 = None
        input_46 = torch.conv2d(
            x_12,
            l_self_modules_stage3_modules_0_modules_branch2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_12 = (
            l_self_modules_stage3_modules_0_modules_branch2_modules_0_parameters_weight_
        ) = None
        input_47 = torch.nn.functional.batch_norm(
            input_46,
            l_self_modules_stage3_modules_0_modules_branch2_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branch2_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branch2_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branch2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_46 = l_self_modules_stage3_modules_0_modules_branch2_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branch2_modules_1_buffers_running_var_ = (
            l_self_modules_stage3_modules_0_modules_branch2_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_0_modules_branch2_modules_1_parameters_bias_
        ) = None
        input_48 = torch.nn.functional.relu(input_47, inplace=True)
        input_47 = None
        input_49 = torch.conv2d(
            input_48,
            l_self_modules_stage3_modules_0_modules_branch2_modules_3_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            116,
        )
        input_48 = (
            l_self_modules_stage3_modules_0_modules_branch2_modules_3_parameters_weight_
        ) = None
        input_50 = torch.nn.functional.batch_norm(
            input_49,
            l_self_modules_stage3_modules_0_modules_branch2_modules_4_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branch2_modules_4_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branch2_modules_4_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branch2_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_49 = l_self_modules_stage3_modules_0_modules_branch2_modules_4_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branch2_modules_4_buffers_running_var_ = (
            l_self_modules_stage3_modules_0_modules_branch2_modules_4_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_0_modules_branch2_modules_4_parameters_bias_
        ) = None
        input_51 = torch.conv2d(
            input_50,
            l_self_modules_stage3_modules_0_modules_branch2_modules_5_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_50 = (
            l_self_modules_stage3_modules_0_modules_branch2_modules_5_parameters_weight_
        ) = None
        input_52 = torch.nn.functional.batch_norm(
            input_51,
            l_self_modules_stage3_modules_0_modules_branch2_modules_6_buffers_running_mean_,
            l_self_modules_stage3_modules_0_modules_branch2_modules_6_buffers_running_var_,
            l_self_modules_stage3_modules_0_modules_branch2_modules_6_parameters_weight_,
            l_self_modules_stage3_modules_0_modules_branch2_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_51 = l_self_modules_stage3_modules_0_modules_branch2_modules_6_buffers_running_mean_ = l_self_modules_stage3_modules_0_modules_branch2_modules_6_buffers_running_var_ = (
            l_self_modules_stage3_modules_0_modules_branch2_modules_6_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_0_modules_branch2_modules_6_parameters_bias_
        ) = None
        input_53 = torch.nn.functional.relu(input_52, inplace=True)
        input_52 = None
        out_4 = torch.cat((input_45, input_53), dim=1)
        input_45 = input_53 = None
        x_13 = out_4.view(1, 2, 116, 14, 14)
        out_4 = None
        transpose_4 = torch.transpose(x_13, 1, 2)
        x_13 = None
        x_14 = transpose_4.contiguous()
        transpose_4 = None
        x_15 = x_14.view(1, 232, 14, 14)
        x_14 = None
        chunk_3 = x_15.chunk(2, dim=1)
        x_15 = None
        x1_3 = chunk_3[0]
        x2_3 = chunk_3[1]
        chunk_3 = None
        input_54 = torch.conv2d(
            x2_3,
            l_self_modules_stage3_modules_1_modules_branch2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x2_3 = (
            l_self_modules_stage3_modules_1_modules_branch2_modules_0_parameters_weight_
        ) = None
        input_55 = torch.nn.functional.batch_norm(
            input_54,
            l_self_modules_stage3_modules_1_modules_branch2_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_branch2_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_branch2_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_branch2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_54 = l_self_modules_stage3_modules_1_modules_branch2_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_branch2_modules_1_buffers_running_var_ = (
            l_self_modules_stage3_modules_1_modules_branch2_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_1_modules_branch2_modules_1_parameters_bias_
        ) = None
        input_56 = torch.nn.functional.relu(input_55, inplace=True)
        input_55 = None
        input_57 = torch.conv2d(
            input_56,
            l_self_modules_stage3_modules_1_modules_branch2_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            116,
        )
        input_56 = (
            l_self_modules_stage3_modules_1_modules_branch2_modules_3_parameters_weight_
        ) = None
        input_58 = torch.nn.functional.batch_norm(
            input_57,
            l_self_modules_stage3_modules_1_modules_branch2_modules_4_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_branch2_modules_4_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_branch2_modules_4_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_branch2_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_57 = l_self_modules_stage3_modules_1_modules_branch2_modules_4_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_branch2_modules_4_buffers_running_var_ = (
            l_self_modules_stage3_modules_1_modules_branch2_modules_4_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_1_modules_branch2_modules_4_parameters_bias_
        ) = None
        input_59 = torch.conv2d(
            input_58,
            l_self_modules_stage3_modules_1_modules_branch2_modules_5_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_58 = (
            l_self_modules_stage3_modules_1_modules_branch2_modules_5_parameters_weight_
        ) = None
        input_60 = torch.nn.functional.batch_norm(
            input_59,
            l_self_modules_stage3_modules_1_modules_branch2_modules_6_buffers_running_mean_,
            l_self_modules_stage3_modules_1_modules_branch2_modules_6_buffers_running_var_,
            l_self_modules_stage3_modules_1_modules_branch2_modules_6_parameters_weight_,
            l_self_modules_stage3_modules_1_modules_branch2_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_59 = l_self_modules_stage3_modules_1_modules_branch2_modules_6_buffers_running_mean_ = l_self_modules_stage3_modules_1_modules_branch2_modules_6_buffers_running_var_ = (
            l_self_modules_stage3_modules_1_modules_branch2_modules_6_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_1_modules_branch2_modules_6_parameters_bias_
        ) = None
        input_61 = torch.nn.functional.relu(input_60, inplace=True)
        input_60 = None
        out_5 = torch.cat((x1_3, input_61), dim=1)
        x1_3 = input_61 = None
        x_16 = out_5.view(1, 2, 116, 14, 14)
        out_5 = None
        transpose_5 = torch.transpose(x_16, 1, 2)
        x_16 = None
        x_17 = transpose_5.contiguous()
        transpose_5 = None
        x_18 = x_17.view(1, 232, 14, 14)
        x_17 = None
        chunk_4 = x_18.chunk(2, dim=1)
        x_18 = None
        x1_4 = chunk_4[0]
        x2_4 = chunk_4[1]
        chunk_4 = None
        input_62 = torch.conv2d(
            x2_4,
            l_self_modules_stage3_modules_2_modules_branch2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x2_4 = (
            l_self_modules_stage3_modules_2_modules_branch2_modules_0_parameters_weight_
        ) = None
        input_63 = torch.nn.functional.batch_norm(
            input_62,
            l_self_modules_stage3_modules_2_modules_branch2_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_branch2_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_branch2_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_branch2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_62 = l_self_modules_stage3_modules_2_modules_branch2_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_branch2_modules_1_buffers_running_var_ = (
            l_self_modules_stage3_modules_2_modules_branch2_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_2_modules_branch2_modules_1_parameters_bias_
        ) = None
        input_64 = torch.nn.functional.relu(input_63, inplace=True)
        input_63 = None
        input_65 = torch.conv2d(
            input_64,
            l_self_modules_stage3_modules_2_modules_branch2_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            116,
        )
        input_64 = (
            l_self_modules_stage3_modules_2_modules_branch2_modules_3_parameters_weight_
        ) = None
        input_66 = torch.nn.functional.batch_norm(
            input_65,
            l_self_modules_stage3_modules_2_modules_branch2_modules_4_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_branch2_modules_4_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_branch2_modules_4_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_branch2_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_65 = l_self_modules_stage3_modules_2_modules_branch2_modules_4_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_branch2_modules_4_buffers_running_var_ = (
            l_self_modules_stage3_modules_2_modules_branch2_modules_4_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_2_modules_branch2_modules_4_parameters_bias_
        ) = None
        input_67 = torch.conv2d(
            input_66,
            l_self_modules_stage3_modules_2_modules_branch2_modules_5_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_66 = (
            l_self_modules_stage3_modules_2_modules_branch2_modules_5_parameters_weight_
        ) = None
        input_68 = torch.nn.functional.batch_norm(
            input_67,
            l_self_modules_stage3_modules_2_modules_branch2_modules_6_buffers_running_mean_,
            l_self_modules_stage3_modules_2_modules_branch2_modules_6_buffers_running_var_,
            l_self_modules_stage3_modules_2_modules_branch2_modules_6_parameters_weight_,
            l_self_modules_stage3_modules_2_modules_branch2_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_67 = l_self_modules_stage3_modules_2_modules_branch2_modules_6_buffers_running_mean_ = l_self_modules_stage3_modules_2_modules_branch2_modules_6_buffers_running_var_ = (
            l_self_modules_stage3_modules_2_modules_branch2_modules_6_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_2_modules_branch2_modules_6_parameters_bias_
        ) = None
        input_69 = torch.nn.functional.relu(input_68, inplace=True)
        input_68 = None
        out_6 = torch.cat((x1_4, input_69), dim=1)
        x1_4 = input_69 = None
        x_19 = out_6.view(1, 2, 116, 14, 14)
        out_6 = None
        transpose_6 = torch.transpose(x_19, 1, 2)
        x_19 = None
        x_20 = transpose_6.contiguous()
        transpose_6 = None
        x_21 = x_20.view(1, 232, 14, 14)
        x_20 = None
        chunk_5 = x_21.chunk(2, dim=1)
        x_21 = None
        x1_5 = chunk_5[0]
        x2_5 = chunk_5[1]
        chunk_5 = None
        input_70 = torch.conv2d(
            x2_5,
            l_self_modules_stage3_modules_3_modules_branch2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x2_5 = (
            l_self_modules_stage3_modules_3_modules_branch2_modules_0_parameters_weight_
        ) = None
        input_71 = torch.nn.functional.batch_norm(
            input_70,
            l_self_modules_stage3_modules_3_modules_branch2_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_3_modules_branch2_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_3_modules_branch2_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_3_modules_branch2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_70 = l_self_modules_stage3_modules_3_modules_branch2_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_3_modules_branch2_modules_1_buffers_running_var_ = (
            l_self_modules_stage3_modules_3_modules_branch2_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_3_modules_branch2_modules_1_parameters_bias_
        ) = None
        input_72 = torch.nn.functional.relu(input_71, inplace=True)
        input_71 = None
        input_73 = torch.conv2d(
            input_72,
            l_self_modules_stage3_modules_3_modules_branch2_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            116,
        )
        input_72 = (
            l_self_modules_stage3_modules_3_modules_branch2_modules_3_parameters_weight_
        ) = None
        input_74 = torch.nn.functional.batch_norm(
            input_73,
            l_self_modules_stage3_modules_3_modules_branch2_modules_4_buffers_running_mean_,
            l_self_modules_stage3_modules_3_modules_branch2_modules_4_buffers_running_var_,
            l_self_modules_stage3_modules_3_modules_branch2_modules_4_parameters_weight_,
            l_self_modules_stage3_modules_3_modules_branch2_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_73 = l_self_modules_stage3_modules_3_modules_branch2_modules_4_buffers_running_mean_ = l_self_modules_stage3_modules_3_modules_branch2_modules_4_buffers_running_var_ = (
            l_self_modules_stage3_modules_3_modules_branch2_modules_4_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_3_modules_branch2_modules_4_parameters_bias_
        ) = None
        input_75 = torch.conv2d(
            input_74,
            l_self_modules_stage3_modules_3_modules_branch2_modules_5_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_74 = (
            l_self_modules_stage3_modules_3_modules_branch2_modules_5_parameters_weight_
        ) = None
        input_76 = torch.nn.functional.batch_norm(
            input_75,
            l_self_modules_stage3_modules_3_modules_branch2_modules_6_buffers_running_mean_,
            l_self_modules_stage3_modules_3_modules_branch2_modules_6_buffers_running_var_,
            l_self_modules_stage3_modules_3_modules_branch2_modules_6_parameters_weight_,
            l_self_modules_stage3_modules_3_modules_branch2_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_75 = l_self_modules_stage3_modules_3_modules_branch2_modules_6_buffers_running_mean_ = l_self_modules_stage3_modules_3_modules_branch2_modules_6_buffers_running_var_ = (
            l_self_modules_stage3_modules_3_modules_branch2_modules_6_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_3_modules_branch2_modules_6_parameters_bias_
        ) = None
        input_77 = torch.nn.functional.relu(input_76, inplace=True)
        input_76 = None
        out_7 = torch.cat((x1_5, input_77), dim=1)
        x1_5 = input_77 = None
        x_22 = out_7.view(1, 2, 116, 14, 14)
        out_7 = None
        transpose_7 = torch.transpose(x_22, 1, 2)
        x_22 = None
        x_23 = transpose_7.contiguous()
        transpose_7 = None
        x_24 = x_23.view(1, 232, 14, 14)
        x_23 = None
        chunk_6 = x_24.chunk(2, dim=1)
        x_24 = None
        x1_6 = chunk_6[0]
        x2_6 = chunk_6[1]
        chunk_6 = None
        input_78 = torch.conv2d(
            x2_6,
            l_self_modules_stage3_modules_4_modules_branch2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x2_6 = (
            l_self_modules_stage3_modules_4_modules_branch2_modules_0_parameters_weight_
        ) = None
        input_79 = torch.nn.functional.batch_norm(
            input_78,
            l_self_modules_stage3_modules_4_modules_branch2_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_4_modules_branch2_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_4_modules_branch2_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_4_modules_branch2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_78 = l_self_modules_stage3_modules_4_modules_branch2_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_4_modules_branch2_modules_1_buffers_running_var_ = (
            l_self_modules_stage3_modules_4_modules_branch2_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_4_modules_branch2_modules_1_parameters_bias_
        ) = None
        input_80 = torch.nn.functional.relu(input_79, inplace=True)
        input_79 = None
        input_81 = torch.conv2d(
            input_80,
            l_self_modules_stage3_modules_4_modules_branch2_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            116,
        )
        input_80 = (
            l_self_modules_stage3_modules_4_modules_branch2_modules_3_parameters_weight_
        ) = None
        input_82 = torch.nn.functional.batch_norm(
            input_81,
            l_self_modules_stage3_modules_4_modules_branch2_modules_4_buffers_running_mean_,
            l_self_modules_stage3_modules_4_modules_branch2_modules_4_buffers_running_var_,
            l_self_modules_stage3_modules_4_modules_branch2_modules_4_parameters_weight_,
            l_self_modules_stage3_modules_4_modules_branch2_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_81 = l_self_modules_stage3_modules_4_modules_branch2_modules_4_buffers_running_mean_ = l_self_modules_stage3_modules_4_modules_branch2_modules_4_buffers_running_var_ = (
            l_self_modules_stage3_modules_4_modules_branch2_modules_4_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_4_modules_branch2_modules_4_parameters_bias_
        ) = None
        input_83 = torch.conv2d(
            input_82,
            l_self_modules_stage3_modules_4_modules_branch2_modules_5_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_82 = (
            l_self_modules_stage3_modules_4_modules_branch2_modules_5_parameters_weight_
        ) = None
        input_84 = torch.nn.functional.batch_norm(
            input_83,
            l_self_modules_stage3_modules_4_modules_branch2_modules_6_buffers_running_mean_,
            l_self_modules_stage3_modules_4_modules_branch2_modules_6_buffers_running_var_,
            l_self_modules_stage3_modules_4_modules_branch2_modules_6_parameters_weight_,
            l_self_modules_stage3_modules_4_modules_branch2_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_83 = l_self_modules_stage3_modules_4_modules_branch2_modules_6_buffers_running_mean_ = l_self_modules_stage3_modules_4_modules_branch2_modules_6_buffers_running_var_ = (
            l_self_modules_stage3_modules_4_modules_branch2_modules_6_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_4_modules_branch2_modules_6_parameters_bias_
        ) = None
        input_85 = torch.nn.functional.relu(input_84, inplace=True)
        input_84 = None
        out_8 = torch.cat((x1_6, input_85), dim=1)
        x1_6 = input_85 = None
        x_25 = out_8.view(1, 2, 116, 14, 14)
        out_8 = None
        transpose_8 = torch.transpose(x_25, 1, 2)
        x_25 = None
        x_26 = transpose_8.contiguous()
        transpose_8 = None
        x_27 = x_26.view(1, 232, 14, 14)
        x_26 = None
        chunk_7 = x_27.chunk(2, dim=1)
        x_27 = None
        x1_7 = chunk_7[0]
        x2_7 = chunk_7[1]
        chunk_7 = None
        input_86 = torch.conv2d(
            x2_7,
            l_self_modules_stage3_modules_5_modules_branch2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x2_7 = (
            l_self_modules_stage3_modules_5_modules_branch2_modules_0_parameters_weight_
        ) = None
        input_87 = torch.nn.functional.batch_norm(
            input_86,
            l_self_modules_stage3_modules_5_modules_branch2_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_5_modules_branch2_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_5_modules_branch2_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_5_modules_branch2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_86 = l_self_modules_stage3_modules_5_modules_branch2_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_5_modules_branch2_modules_1_buffers_running_var_ = (
            l_self_modules_stage3_modules_5_modules_branch2_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_5_modules_branch2_modules_1_parameters_bias_
        ) = None
        input_88 = torch.nn.functional.relu(input_87, inplace=True)
        input_87 = None
        input_89 = torch.conv2d(
            input_88,
            l_self_modules_stage3_modules_5_modules_branch2_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            116,
        )
        input_88 = (
            l_self_modules_stage3_modules_5_modules_branch2_modules_3_parameters_weight_
        ) = None
        input_90 = torch.nn.functional.batch_norm(
            input_89,
            l_self_modules_stage3_modules_5_modules_branch2_modules_4_buffers_running_mean_,
            l_self_modules_stage3_modules_5_modules_branch2_modules_4_buffers_running_var_,
            l_self_modules_stage3_modules_5_modules_branch2_modules_4_parameters_weight_,
            l_self_modules_stage3_modules_5_modules_branch2_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_89 = l_self_modules_stage3_modules_5_modules_branch2_modules_4_buffers_running_mean_ = l_self_modules_stage3_modules_5_modules_branch2_modules_4_buffers_running_var_ = (
            l_self_modules_stage3_modules_5_modules_branch2_modules_4_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_5_modules_branch2_modules_4_parameters_bias_
        ) = None
        input_91 = torch.conv2d(
            input_90,
            l_self_modules_stage3_modules_5_modules_branch2_modules_5_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_90 = (
            l_self_modules_stage3_modules_5_modules_branch2_modules_5_parameters_weight_
        ) = None
        input_92 = torch.nn.functional.batch_norm(
            input_91,
            l_self_modules_stage3_modules_5_modules_branch2_modules_6_buffers_running_mean_,
            l_self_modules_stage3_modules_5_modules_branch2_modules_6_buffers_running_var_,
            l_self_modules_stage3_modules_5_modules_branch2_modules_6_parameters_weight_,
            l_self_modules_stage3_modules_5_modules_branch2_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_91 = l_self_modules_stage3_modules_5_modules_branch2_modules_6_buffers_running_mean_ = l_self_modules_stage3_modules_5_modules_branch2_modules_6_buffers_running_var_ = (
            l_self_modules_stage3_modules_5_modules_branch2_modules_6_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_5_modules_branch2_modules_6_parameters_bias_
        ) = None
        input_93 = torch.nn.functional.relu(input_92, inplace=True)
        input_92 = None
        out_9 = torch.cat((x1_7, input_93), dim=1)
        x1_7 = input_93 = None
        x_28 = out_9.view(1, 2, 116, 14, 14)
        out_9 = None
        transpose_9 = torch.transpose(x_28, 1, 2)
        x_28 = None
        x_29 = transpose_9.contiguous()
        transpose_9 = None
        x_30 = x_29.view(1, 232, 14, 14)
        x_29 = None
        chunk_8 = x_30.chunk(2, dim=1)
        x_30 = None
        x1_8 = chunk_8[0]
        x2_8 = chunk_8[1]
        chunk_8 = None
        input_94 = torch.conv2d(
            x2_8,
            l_self_modules_stage3_modules_6_modules_branch2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x2_8 = (
            l_self_modules_stage3_modules_6_modules_branch2_modules_0_parameters_weight_
        ) = None
        input_95 = torch.nn.functional.batch_norm(
            input_94,
            l_self_modules_stage3_modules_6_modules_branch2_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_6_modules_branch2_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_6_modules_branch2_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_6_modules_branch2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_94 = l_self_modules_stage3_modules_6_modules_branch2_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_6_modules_branch2_modules_1_buffers_running_var_ = (
            l_self_modules_stage3_modules_6_modules_branch2_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_6_modules_branch2_modules_1_parameters_bias_
        ) = None
        input_96 = torch.nn.functional.relu(input_95, inplace=True)
        input_95 = None
        input_97 = torch.conv2d(
            input_96,
            l_self_modules_stage3_modules_6_modules_branch2_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            116,
        )
        input_96 = (
            l_self_modules_stage3_modules_6_modules_branch2_modules_3_parameters_weight_
        ) = None
        input_98 = torch.nn.functional.batch_norm(
            input_97,
            l_self_modules_stage3_modules_6_modules_branch2_modules_4_buffers_running_mean_,
            l_self_modules_stage3_modules_6_modules_branch2_modules_4_buffers_running_var_,
            l_self_modules_stage3_modules_6_modules_branch2_modules_4_parameters_weight_,
            l_self_modules_stage3_modules_6_modules_branch2_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_97 = l_self_modules_stage3_modules_6_modules_branch2_modules_4_buffers_running_mean_ = l_self_modules_stage3_modules_6_modules_branch2_modules_4_buffers_running_var_ = (
            l_self_modules_stage3_modules_6_modules_branch2_modules_4_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_6_modules_branch2_modules_4_parameters_bias_
        ) = None
        input_99 = torch.conv2d(
            input_98,
            l_self_modules_stage3_modules_6_modules_branch2_modules_5_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_98 = (
            l_self_modules_stage3_modules_6_modules_branch2_modules_5_parameters_weight_
        ) = None
        input_100 = torch.nn.functional.batch_norm(
            input_99,
            l_self_modules_stage3_modules_6_modules_branch2_modules_6_buffers_running_mean_,
            l_self_modules_stage3_modules_6_modules_branch2_modules_6_buffers_running_var_,
            l_self_modules_stage3_modules_6_modules_branch2_modules_6_parameters_weight_,
            l_self_modules_stage3_modules_6_modules_branch2_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_99 = l_self_modules_stage3_modules_6_modules_branch2_modules_6_buffers_running_mean_ = l_self_modules_stage3_modules_6_modules_branch2_modules_6_buffers_running_var_ = (
            l_self_modules_stage3_modules_6_modules_branch2_modules_6_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_6_modules_branch2_modules_6_parameters_bias_
        ) = None
        input_101 = torch.nn.functional.relu(input_100, inplace=True)
        input_100 = None
        out_10 = torch.cat((x1_8, input_101), dim=1)
        x1_8 = input_101 = None
        x_31 = out_10.view(1, 2, 116, 14, 14)
        out_10 = None
        transpose_10 = torch.transpose(x_31, 1, 2)
        x_31 = None
        x_32 = transpose_10.contiguous()
        transpose_10 = None
        x_33 = x_32.view(1, 232, 14, 14)
        x_32 = None
        chunk_9 = x_33.chunk(2, dim=1)
        x_33 = None
        x1_9 = chunk_9[0]
        x2_9 = chunk_9[1]
        chunk_9 = None
        input_102 = torch.conv2d(
            x2_9,
            l_self_modules_stage3_modules_7_modules_branch2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x2_9 = (
            l_self_modules_stage3_modules_7_modules_branch2_modules_0_parameters_weight_
        ) = None
        input_103 = torch.nn.functional.batch_norm(
            input_102,
            l_self_modules_stage3_modules_7_modules_branch2_modules_1_buffers_running_mean_,
            l_self_modules_stage3_modules_7_modules_branch2_modules_1_buffers_running_var_,
            l_self_modules_stage3_modules_7_modules_branch2_modules_1_parameters_weight_,
            l_self_modules_stage3_modules_7_modules_branch2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_102 = l_self_modules_stage3_modules_7_modules_branch2_modules_1_buffers_running_mean_ = l_self_modules_stage3_modules_7_modules_branch2_modules_1_buffers_running_var_ = (
            l_self_modules_stage3_modules_7_modules_branch2_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_7_modules_branch2_modules_1_parameters_bias_
        ) = None
        input_104 = torch.nn.functional.relu(input_103, inplace=True)
        input_103 = None
        input_105 = torch.conv2d(
            input_104,
            l_self_modules_stage3_modules_7_modules_branch2_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            116,
        )
        input_104 = (
            l_self_modules_stage3_modules_7_modules_branch2_modules_3_parameters_weight_
        ) = None
        input_106 = torch.nn.functional.batch_norm(
            input_105,
            l_self_modules_stage3_modules_7_modules_branch2_modules_4_buffers_running_mean_,
            l_self_modules_stage3_modules_7_modules_branch2_modules_4_buffers_running_var_,
            l_self_modules_stage3_modules_7_modules_branch2_modules_4_parameters_weight_,
            l_self_modules_stage3_modules_7_modules_branch2_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_105 = l_self_modules_stage3_modules_7_modules_branch2_modules_4_buffers_running_mean_ = l_self_modules_stage3_modules_7_modules_branch2_modules_4_buffers_running_var_ = (
            l_self_modules_stage3_modules_7_modules_branch2_modules_4_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_7_modules_branch2_modules_4_parameters_bias_
        ) = None
        input_107 = torch.conv2d(
            input_106,
            l_self_modules_stage3_modules_7_modules_branch2_modules_5_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_106 = (
            l_self_modules_stage3_modules_7_modules_branch2_modules_5_parameters_weight_
        ) = None
        input_108 = torch.nn.functional.batch_norm(
            input_107,
            l_self_modules_stage3_modules_7_modules_branch2_modules_6_buffers_running_mean_,
            l_self_modules_stage3_modules_7_modules_branch2_modules_6_buffers_running_var_,
            l_self_modules_stage3_modules_7_modules_branch2_modules_6_parameters_weight_,
            l_self_modules_stage3_modules_7_modules_branch2_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_107 = l_self_modules_stage3_modules_7_modules_branch2_modules_6_buffers_running_mean_ = l_self_modules_stage3_modules_7_modules_branch2_modules_6_buffers_running_var_ = (
            l_self_modules_stage3_modules_7_modules_branch2_modules_6_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_7_modules_branch2_modules_6_parameters_bias_
        ) = None
        input_109 = torch.nn.functional.relu(input_108, inplace=True)
        input_108 = None
        out_11 = torch.cat((x1_9, input_109), dim=1)
        x1_9 = input_109 = None
        x_34 = out_11.view(1, 2, 116, 14, 14)
        out_11 = None
        transpose_11 = torch.transpose(x_34, 1, 2)
        x_34 = None
        x_35 = transpose_11.contiguous()
        transpose_11 = None
        x_36 = x_35.view(1, 232, 14, 14)
        x_35 = None
        input_110 = torch.conv2d(
            x_36,
            l_self_modules_stage4_modules_0_modules_branch1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            232,
        )
        l_self_modules_stage4_modules_0_modules_branch1_modules_0_parameters_weight_ = (
            None
        )
        input_111 = torch.nn.functional.batch_norm(
            input_110,
            l_self_modules_stage4_modules_0_modules_branch1_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branch1_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branch1_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branch1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_110 = l_self_modules_stage4_modules_0_modules_branch1_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branch1_modules_1_buffers_running_var_ = (
            l_self_modules_stage4_modules_0_modules_branch1_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_0_modules_branch1_modules_1_parameters_bias_
        ) = None
        input_112 = torch.conv2d(
            input_111,
            l_self_modules_stage4_modules_0_modules_branch1_modules_2_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_111 = (
            l_self_modules_stage4_modules_0_modules_branch1_modules_2_parameters_weight_
        ) = None
        input_113 = torch.nn.functional.batch_norm(
            input_112,
            l_self_modules_stage4_modules_0_modules_branch1_modules_3_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branch1_modules_3_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branch1_modules_3_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branch1_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_112 = l_self_modules_stage4_modules_0_modules_branch1_modules_3_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branch1_modules_3_buffers_running_var_ = (
            l_self_modules_stage4_modules_0_modules_branch1_modules_3_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_0_modules_branch1_modules_3_parameters_bias_
        ) = None
        input_114 = torch.nn.functional.relu(input_113, inplace=True)
        input_113 = None
        input_115 = torch.conv2d(
            x_36,
            l_self_modules_stage4_modules_0_modules_branch2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_36 = (
            l_self_modules_stage4_modules_0_modules_branch2_modules_0_parameters_weight_
        ) = None
        input_116 = torch.nn.functional.batch_norm(
            input_115,
            l_self_modules_stage4_modules_0_modules_branch2_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branch2_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branch2_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branch2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_115 = l_self_modules_stage4_modules_0_modules_branch2_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branch2_modules_1_buffers_running_var_ = (
            l_self_modules_stage4_modules_0_modules_branch2_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_0_modules_branch2_modules_1_parameters_bias_
        ) = None
        input_117 = torch.nn.functional.relu(input_116, inplace=True)
        input_116 = None
        input_118 = torch.conv2d(
            input_117,
            l_self_modules_stage4_modules_0_modules_branch2_modules_3_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            232,
        )
        input_117 = (
            l_self_modules_stage4_modules_0_modules_branch2_modules_3_parameters_weight_
        ) = None
        input_119 = torch.nn.functional.batch_norm(
            input_118,
            l_self_modules_stage4_modules_0_modules_branch2_modules_4_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branch2_modules_4_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branch2_modules_4_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branch2_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_118 = l_self_modules_stage4_modules_0_modules_branch2_modules_4_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branch2_modules_4_buffers_running_var_ = (
            l_self_modules_stage4_modules_0_modules_branch2_modules_4_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_0_modules_branch2_modules_4_parameters_bias_
        ) = None
        input_120 = torch.conv2d(
            input_119,
            l_self_modules_stage4_modules_0_modules_branch2_modules_5_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_119 = (
            l_self_modules_stage4_modules_0_modules_branch2_modules_5_parameters_weight_
        ) = None
        input_121 = torch.nn.functional.batch_norm(
            input_120,
            l_self_modules_stage4_modules_0_modules_branch2_modules_6_buffers_running_mean_,
            l_self_modules_stage4_modules_0_modules_branch2_modules_6_buffers_running_var_,
            l_self_modules_stage4_modules_0_modules_branch2_modules_6_parameters_weight_,
            l_self_modules_stage4_modules_0_modules_branch2_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_120 = l_self_modules_stage4_modules_0_modules_branch2_modules_6_buffers_running_mean_ = l_self_modules_stage4_modules_0_modules_branch2_modules_6_buffers_running_var_ = (
            l_self_modules_stage4_modules_0_modules_branch2_modules_6_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_0_modules_branch2_modules_6_parameters_bias_
        ) = None
        input_122 = torch.nn.functional.relu(input_121, inplace=True)
        input_121 = None
        out_12 = torch.cat((input_114, input_122), dim=1)
        input_114 = input_122 = None
        x_37 = out_12.view(1, 2, 232, 7, 7)
        out_12 = None
        transpose_12 = torch.transpose(x_37, 1, 2)
        x_37 = None
        x_38 = transpose_12.contiguous()
        transpose_12 = None
        x_39 = x_38.view(1, 464, 7, 7)
        x_38 = None
        chunk_10 = x_39.chunk(2, dim=1)
        x_39 = None
        x1_10 = chunk_10[0]
        x2_10 = chunk_10[1]
        chunk_10 = None
        input_123 = torch.conv2d(
            x2_10,
            l_self_modules_stage4_modules_1_modules_branch2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x2_10 = (
            l_self_modules_stage4_modules_1_modules_branch2_modules_0_parameters_weight_
        ) = None
        input_124 = torch.nn.functional.batch_norm(
            input_123,
            l_self_modules_stage4_modules_1_modules_branch2_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branch2_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branch2_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branch2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_123 = l_self_modules_stage4_modules_1_modules_branch2_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branch2_modules_1_buffers_running_var_ = (
            l_self_modules_stage4_modules_1_modules_branch2_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_1_modules_branch2_modules_1_parameters_bias_
        ) = None
        input_125 = torch.nn.functional.relu(input_124, inplace=True)
        input_124 = None
        input_126 = torch.conv2d(
            input_125,
            l_self_modules_stage4_modules_1_modules_branch2_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            232,
        )
        input_125 = (
            l_self_modules_stage4_modules_1_modules_branch2_modules_3_parameters_weight_
        ) = None
        input_127 = torch.nn.functional.batch_norm(
            input_126,
            l_self_modules_stage4_modules_1_modules_branch2_modules_4_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branch2_modules_4_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branch2_modules_4_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branch2_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_126 = l_self_modules_stage4_modules_1_modules_branch2_modules_4_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branch2_modules_4_buffers_running_var_ = (
            l_self_modules_stage4_modules_1_modules_branch2_modules_4_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_1_modules_branch2_modules_4_parameters_bias_
        ) = None
        input_128 = torch.conv2d(
            input_127,
            l_self_modules_stage4_modules_1_modules_branch2_modules_5_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_127 = (
            l_self_modules_stage4_modules_1_modules_branch2_modules_5_parameters_weight_
        ) = None
        input_129 = torch.nn.functional.batch_norm(
            input_128,
            l_self_modules_stage4_modules_1_modules_branch2_modules_6_buffers_running_mean_,
            l_self_modules_stage4_modules_1_modules_branch2_modules_6_buffers_running_var_,
            l_self_modules_stage4_modules_1_modules_branch2_modules_6_parameters_weight_,
            l_self_modules_stage4_modules_1_modules_branch2_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_128 = l_self_modules_stage4_modules_1_modules_branch2_modules_6_buffers_running_mean_ = l_self_modules_stage4_modules_1_modules_branch2_modules_6_buffers_running_var_ = (
            l_self_modules_stage4_modules_1_modules_branch2_modules_6_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_1_modules_branch2_modules_6_parameters_bias_
        ) = None
        input_130 = torch.nn.functional.relu(input_129, inplace=True)
        input_129 = None
        out_13 = torch.cat((x1_10, input_130), dim=1)
        x1_10 = input_130 = None
        x_40 = out_13.view(1, 2, 232, 7, 7)
        out_13 = None
        transpose_13 = torch.transpose(x_40, 1, 2)
        x_40 = None
        x_41 = transpose_13.contiguous()
        transpose_13 = None
        x_42 = x_41.view(1, 464, 7, 7)
        x_41 = None
        chunk_11 = x_42.chunk(2, dim=1)
        x_42 = None
        x1_11 = chunk_11[0]
        x2_11 = chunk_11[1]
        chunk_11 = None
        input_131 = torch.conv2d(
            x2_11,
            l_self_modules_stage4_modules_2_modules_branch2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x2_11 = (
            l_self_modules_stage4_modules_2_modules_branch2_modules_0_parameters_weight_
        ) = None
        input_132 = torch.nn.functional.batch_norm(
            input_131,
            l_self_modules_stage4_modules_2_modules_branch2_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_2_modules_branch2_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_2_modules_branch2_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_2_modules_branch2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_131 = l_self_modules_stage4_modules_2_modules_branch2_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_2_modules_branch2_modules_1_buffers_running_var_ = (
            l_self_modules_stage4_modules_2_modules_branch2_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_2_modules_branch2_modules_1_parameters_bias_
        ) = None
        input_133 = torch.nn.functional.relu(input_132, inplace=True)
        input_132 = None
        input_134 = torch.conv2d(
            input_133,
            l_self_modules_stage4_modules_2_modules_branch2_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            232,
        )
        input_133 = (
            l_self_modules_stage4_modules_2_modules_branch2_modules_3_parameters_weight_
        ) = None
        input_135 = torch.nn.functional.batch_norm(
            input_134,
            l_self_modules_stage4_modules_2_modules_branch2_modules_4_buffers_running_mean_,
            l_self_modules_stage4_modules_2_modules_branch2_modules_4_buffers_running_var_,
            l_self_modules_stage4_modules_2_modules_branch2_modules_4_parameters_weight_,
            l_self_modules_stage4_modules_2_modules_branch2_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_134 = l_self_modules_stage4_modules_2_modules_branch2_modules_4_buffers_running_mean_ = l_self_modules_stage4_modules_2_modules_branch2_modules_4_buffers_running_var_ = (
            l_self_modules_stage4_modules_2_modules_branch2_modules_4_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_2_modules_branch2_modules_4_parameters_bias_
        ) = None
        input_136 = torch.conv2d(
            input_135,
            l_self_modules_stage4_modules_2_modules_branch2_modules_5_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_135 = (
            l_self_modules_stage4_modules_2_modules_branch2_modules_5_parameters_weight_
        ) = None
        input_137 = torch.nn.functional.batch_norm(
            input_136,
            l_self_modules_stage4_modules_2_modules_branch2_modules_6_buffers_running_mean_,
            l_self_modules_stage4_modules_2_modules_branch2_modules_6_buffers_running_var_,
            l_self_modules_stage4_modules_2_modules_branch2_modules_6_parameters_weight_,
            l_self_modules_stage4_modules_2_modules_branch2_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_136 = l_self_modules_stage4_modules_2_modules_branch2_modules_6_buffers_running_mean_ = l_self_modules_stage4_modules_2_modules_branch2_modules_6_buffers_running_var_ = (
            l_self_modules_stage4_modules_2_modules_branch2_modules_6_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_2_modules_branch2_modules_6_parameters_bias_
        ) = None
        input_138 = torch.nn.functional.relu(input_137, inplace=True)
        input_137 = None
        out_14 = torch.cat((x1_11, input_138), dim=1)
        x1_11 = input_138 = None
        x_43 = out_14.view(1, 2, 232, 7, 7)
        out_14 = None
        transpose_14 = torch.transpose(x_43, 1, 2)
        x_43 = None
        x_44 = transpose_14.contiguous()
        transpose_14 = None
        x_45 = x_44.view(1, 464, 7, 7)
        x_44 = None
        chunk_12 = x_45.chunk(2, dim=1)
        x_45 = None
        x1_12 = chunk_12[0]
        x2_12 = chunk_12[1]
        chunk_12 = None
        input_139 = torch.conv2d(
            x2_12,
            l_self_modules_stage4_modules_3_modules_branch2_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x2_12 = (
            l_self_modules_stage4_modules_3_modules_branch2_modules_0_parameters_weight_
        ) = None
        input_140 = torch.nn.functional.batch_norm(
            input_139,
            l_self_modules_stage4_modules_3_modules_branch2_modules_1_buffers_running_mean_,
            l_self_modules_stage4_modules_3_modules_branch2_modules_1_buffers_running_var_,
            l_self_modules_stage4_modules_3_modules_branch2_modules_1_parameters_weight_,
            l_self_modules_stage4_modules_3_modules_branch2_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_139 = l_self_modules_stage4_modules_3_modules_branch2_modules_1_buffers_running_mean_ = l_self_modules_stage4_modules_3_modules_branch2_modules_1_buffers_running_var_ = (
            l_self_modules_stage4_modules_3_modules_branch2_modules_1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_3_modules_branch2_modules_1_parameters_bias_
        ) = None
        input_141 = torch.nn.functional.relu(input_140, inplace=True)
        input_140 = None
        input_142 = torch.conv2d(
            input_141,
            l_self_modules_stage4_modules_3_modules_branch2_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            232,
        )
        input_141 = (
            l_self_modules_stage4_modules_3_modules_branch2_modules_3_parameters_weight_
        ) = None
        input_143 = torch.nn.functional.batch_norm(
            input_142,
            l_self_modules_stage4_modules_3_modules_branch2_modules_4_buffers_running_mean_,
            l_self_modules_stage4_modules_3_modules_branch2_modules_4_buffers_running_var_,
            l_self_modules_stage4_modules_3_modules_branch2_modules_4_parameters_weight_,
            l_self_modules_stage4_modules_3_modules_branch2_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_142 = l_self_modules_stage4_modules_3_modules_branch2_modules_4_buffers_running_mean_ = l_self_modules_stage4_modules_3_modules_branch2_modules_4_buffers_running_var_ = (
            l_self_modules_stage4_modules_3_modules_branch2_modules_4_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_3_modules_branch2_modules_4_parameters_bias_
        ) = None
        input_144 = torch.conv2d(
            input_143,
            l_self_modules_stage4_modules_3_modules_branch2_modules_5_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_143 = (
            l_self_modules_stage4_modules_3_modules_branch2_modules_5_parameters_weight_
        ) = None
        input_145 = torch.nn.functional.batch_norm(
            input_144,
            l_self_modules_stage4_modules_3_modules_branch2_modules_6_buffers_running_mean_,
            l_self_modules_stage4_modules_3_modules_branch2_modules_6_buffers_running_var_,
            l_self_modules_stage4_modules_3_modules_branch2_modules_6_parameters_weight_,
            l_self_modules_stage4_modules_3_modules_branch2_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_144 = l_self_modules_stage4_modules_3_modules_branch2_modules_6_buffers_running_mean_ = l_self_modules_stage4_modules_3_modules_branch2_modules_6_buffers_running_var_ = (
            l_self_modules_stage4_modules_3_modules_branch2_modules_6_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_3_modules_branch2_modules_6_parameters_bias_
        ) = None
        input_146 = torch.nn.functional.relu(input_145, inplace=True)
        input_145 = None
        out_15 = torch.cat((x1_12, input_146), dim=1)
        x1_12 = input_146 = None
        x_46 = out_15.view(1, 2, 232, 7, 7)
        out_15 = None
        transpose_15 = torch.transpose(x_46, 1, 2)
        x_46 = None
        x_47 = transpose_15.contiguous()
        transpose_15 = None
        x_48 = x_47.view(1, 464, 7, 7)
        x_47 = None
        input_147 = torch.conv2d(
            x_48,
            l_self_modules_conv5_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_48 = l_self_modules_conv5_modules_0_parameters_weight_ = None
        input_148 = torch.nn.functional.batch_norm(
            input_147,
            l_self_modules_conv5_modules_1_buffers_running_mean_,
            l_self_modules_conv5_modules_1_buffers_running_var_,
            l_self_modules_conv5_modules_1_parameters_weight_,
            l_self_modules_conv5_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_147 = (
            l_self_modules_conv5_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_conv5_modules_1_buffers_running_var_
        ) = (
            l_self_modules_conv5_modules_1_parameters_weight_
        ) = l_self_modules_conv5_modules_1_parameters_bias_ = None
        input_149 = torch.nn.functional.relu(input_148, inplace=True)
        input_148 = None
        x_49 = input_149.mean([2, 3])
        input_149 = None
        x_50 = torch._C._nn.linear(
            x_49,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_49 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_50,)
