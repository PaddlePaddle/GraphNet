import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_conv_down_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_conv_transition_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_conv_transition_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_conv_transition_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_conv_transition_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_conv_transition_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_conv_down_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_conv_transition_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_conv_transition_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_conv_transition_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_conv_transition_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_conv_transition_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_conv_down_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_conv_transition_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_conv_transition_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_conv_transition_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_conv_transition_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_conv_transition_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_conv_down_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_conv_exp_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_conv_exp_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_conv_exp_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_conv_exp_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_conv_exp_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_conv_transition_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_conv_transition_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_conv_transition_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_conv_transition_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_conv_transition_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv1_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_stem_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_conv_down_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_conv_down_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_conv_exp_modules_bn_parameters_bias_ = (
            L_self_modules_stages_modules_0_modules_conv_exp_modules_bn_parameters_bias_
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_conv_transition_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_conv_transition_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_conv_transition_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_conv_transition_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_conv_transition_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_conv_transition_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_conv_down_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_conv_down_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_parameters_bias_ = (
            L_self_modules_stages_modules_1_modules_conv_exp_modules_bn_parameters_bias_
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_conv_transition_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_conv_transition_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_conv_transition_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_conv_transition_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_conv_transition_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_conv_transition_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_conv_down_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_conv_down_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_parameters_bias_ = (
            L_self_modules_stages_modules_2_modules_conv_exp_modules_bn_parameters_bias_
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_conv_transition_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_conv_transition_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_conv_transition_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_conv_transition_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_conv_transition_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_conv_transition_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_conv_down_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_conv_down_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_conv_exp_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_conv_exp_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_conv_exp_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_conv_exp_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_conv_exp_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_parameters_bias_ = (
            L_self_modules_stages_modules_3_modules_conv_exp_modules_bn_parameters_bias_
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_conv_transition_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_conv_transition_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_conv_transition_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_conv_transition_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_conv_transition_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_conv_transition_modules_bn_parameters_bias_
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_conv1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_modules_conv1_modules_conv_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_conv1_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_conv1_modules_bn_parameters_bias_ = None
        x_2 = torch.nn.functional.silu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_stem_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_stem_modules_conv2_modules_conv_parameters_weight_ = None
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = (
            l_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_conv2_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_conv2_modules_bn_parameters_bias_ = None
        x_5 = torch.nn.functional.silu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_stages_modules_0_modules_conv_down_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_stages_modules_0_modules_conv_down_modules_conv_parameters_weight_ = (None)
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = l_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_conv_down_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_conv_down_modules_bn_parameters_bias_ = (None)
        x_8 = torch.nn.functional.silu(x_7, inplace=True)
        x_7 = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_stages_modules_0_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_8 = l_self_modules_stages_modules_0_modules_conv_exp_modules_conv_parameters_weight_ = (None)
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_stages_modules_0_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = l_self_modules_stages_modules_0_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_conv_exp_modules_bn_parameters_weight_ = (
            l_self_modules_stages_modules_0_modules_conv_exp_modules_bn_parameters_bias_
        ) = None
        x_11 = torch.nn.functional.silu(x_10, inplace=True)
        x_10 = None
        split = x_11.split(80, dim=1)
        x_11 = None
        x1 = split[0]
        x2 = split[1]
        split = None
        x_12 = torch.conv2d(
            x1,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_12 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_14 = torch.nn.functional.silu(x_13, inplace=True)
        x_13 = None
        x_15 = torch.conv2d(
            x_14,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_14 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_15 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_17 = torch.nn.functional.silu(x_16, inplace=True)
        x_16 = None
        x_18 = x_17 + x1
        x_17 = x1 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_21 = torch.nn.functional.silu(x_20, inplace=True)
        x_20 = None
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_21 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_24 = torch.nn.functional.silu(x_23, inplace=True)
        x_23 = None
        x_25 = x_24 + x_18
        x_24 = x_18 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_28 = torch.nn.functional.silu(x_27, inplace=True)
        x_27 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_31 = torch.nn.functional.silu(x_30, inplace=True)
        x_30 = None
        x_32 = x_31 + x_25
        x_31 = x_25 = None
        cat = torch.cat([x_32, x2], dim=1)
        x_32 = x2 = None
        x_33 = torch.conv2d(
            cat,
            l_self_modules_stages_modules_0_modules_conv_transition_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat = l_self_modules_stages_modules_0_modules_conv_transition_modules_conv_parameters_weight_ = (None)
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_parameters_bias_ = (None)
        x_35 = torch.nn.functional.silu(x_34, inplace=True)
        x_34 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_stages_modules_1_modules_conv_down_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_stages_modules_1_modules_conv_down_modules_conv_parameters_weight_ = (None)
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_bias_ = (None)
        x_38 = torch.nn.functional.silu(x_37, inplace=True)
        x_37 = None
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_stages_modules_1_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_38 = l_self_modules_stages_modules_1_modules_conv_exp_modules_conv_parameters_weight_ = (None)
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_39 = l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_parameters_bias_
        ) = None
        x_41 = torch.nn.functional.silu(x_40, inplace=True)
        x_40 = None
        split_1 = x_41.split(160, dim=1)
        x_41 = None
        x1_1 = split_1[0]
        x2_1 = split_1[1]
        split_1 = None
        x_42 = torch.conv2d(
            x1_1,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_44 = torch.nn.functional.silu(x_43, inplace=True)
        x_43 = None
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_44 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_47 = torch.nn.functional.silu(x_46, inplace=True)
        x_46 = None
        x_48 = x_47 + x1_1
        x_47 = x1_1 = None
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_49 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_51 = torch.nn.functional.silu(x_50, inplace=True)
        x_50 = None
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_51 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_54 = torch.nn.functional.silu(x_53, inplace=True)
        x_53 = None
        x_55 = x_54 + x_48
        x_54 = x_48 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_58 = torch.nn.functional.silu(x_57, inplace=True)
        x_57 = None
        x_59 = torch.conv2d(
            x_58,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_58 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_59 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_61 = torch.nn.functional.silu(x_60, inplace=True)
        x_60 = None
        x_62 = x_61 + x_55
        x_61 = x_55 = None
        x_63 = torch.conv2d(
            x_62,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_65 = torch.nn.functional.silu(x_64, inplace=True)
        x_64 = None
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_68 = torch.nn.functional.silu(x_67, inplace=True)
        x_67 = None
        x_69 = x_68 + x_62
        x_68 = x_62 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_72 = torch.nn.functional.silu(x_71, inplace=True)
        x_71 = None
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_72 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_75 = torch.nn.functional.silu(x_74, inplace=True)
        x_74 = None
        x_76 = x_75 + x_69
        x_75 = x_69 = None
        x_77 = torch.conv2d(
            x_76,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_77 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_79 = torch.nn.functional.silu(x_78, inplace=True)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_82 = torch.nn.functional.silu(x_81, inplace=True)
        x_81 = None
        x_83 = x_82 + x_76
        x_82 = x_76 = None
        x_84 = torch.conv2d(
            x_83,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_85 = torch.nn.functional.batch_norm(
            x_84,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_84 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_86 = torch.nn.functional.silu(x_85, inplace=True)
        x_85 = None
        x_87 = torch.conv2d(
            x_86,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_86 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_88 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_87 = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_89 = torch.nn.functional.silu(x_88, inplace=True)
        x_88 = None
        x_90 = x_89 + x_83
        x_89 = x_83 = None
        cat_1 = torch.cat([x_90, x2_1], dim=1)
        x_90 = x2_1 = None
        x_91 = torch.conv2d(
            cat_1,
            l_self_modules_stages_modules_1_modules_conv_transition_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_1 = l_self_modules_stages_modules_1_modules_conv_transition_modules_conv_parameters_weight_ = (None)
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_parameters_bias_ = (None)
        x_93 = torch.nn.functional.silu(x_92, inplace=True)
        x_92 = None
        x_94 = torch.conv2d(
            x_93,
            l_self_modules_stages_modules_2_modules_conv_down_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_93 = l_self_modules_stages_modules_2_modules_conv_down_modules_conv_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_bias_ = (None)
        x_96 = torch.nn.functional.silu(x_95, inplace=True)
        x_95 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_stages_modules_2_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_96 = l_self_modules_stages_modules_2_modules_conv_exp_modules_conv_parameters_weight_ = (None)
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_97 = l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_parameters_bias_
        ) = None
        x_99 = torch.nn.functional.silu(x_98, inplace=True)
        x_98 = None
        split_2 = x_99.split(320, dim=1)
        x_99 = None
        x1_2 = split_2[0]
        x2_2 = split_2[1]
        split_2 = None
        x_100 = torch.conv2d(
            x1_2,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_100 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_102 = torch.nn.functional.silu(x_101, inplace=True)
        x_101 = None
        x_103 = torch.conv2d(
            x_102,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_102 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_104 = torch.nn.functional.batch_norm(
            x_103,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_103 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_105 = torch.nn.functional.silu(x_104, inplace=True)
        x_104 = None
        x_106 = x_105 + x1_2
        x_105 = x1_2 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_109 = torch.nn.functional.silu(x_108, inplace=True)
        x_108 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_112 = torch.nn.functional.silu(x_111, inplace=True)
        x_111 = None
        x_113 = x_112 + x_106
        x_112 = x_106 = None
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_115 = torch.nn.functional.batch_norm(
            x_114,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_114 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_116 = torch.nn.functional.silu(x_115, inplace=True)
        x_115 = None
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_116 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_117 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_119 = torch.nn.functional.silu(x_118, inplace=True)
        x_118 = None
        x_120 = x_119 + x_113
        x_119 = x_113 = None
        x_121 = torch.conv2d(
            x_120,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_121 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_123 = torch.nn.functional.silu(x_122, inplace=True)
        x_122 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_123 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_124 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_126 = torch.nn.functional.silu(x_125, inplace=True)
        x_125 = None
        x_127 = x_126 + x_120
        x_126 = x_120 = None
        x_128 = torch.conv2d(
            x_127,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_130 = torch.nn.functional.silu(x_129, inplace=True)
        x_129 = None
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_130 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_133 = torch.nn.functional.silu(x_132, inplace=True)
        x_132 = None
        x_134 = x_133 + x_127
        x_133 = x_127 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_136 = torch.nn.functional.batch_norm(
            x_135,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_135 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_137 = torch.nn.functional.silu(x_136, inplace=True)
        x_136 = None
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_137 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_138 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_140 = torch.nn.functional.silu(x_139, inplace=True)
        x_139 = None
        x_141 = x_140 + x_134
        x_140 = x_134 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_142 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_144 = torch.nn.functional.silu(x_143, inplace=True)
        x_143 = None
        x_145 = torch.conv2d(
            x_144,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_144 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_146 = torch.nn.functional.batch_norm(
            x_145,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_145 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_147 = torch.nn.functional.silu(x_146, inplace=True)
        x_146 = None
        x_148 = x_147 + x_141
        x_147 = x_141 = None
        x_149 = torch.conv2d(
            x_148,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_150 = torch.nn.functional.batch_norm(
            x_149,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_149 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_151 = torch.nn.functional.silu(x_150, inplace=True)
        x_150 = None
        x_152 = torch.conv2d(
            x_151,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_151 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_152 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_154 = torch.nn.functional.silu(x_153, inplace=True)
        x_153 = None
        x_155 = x_154 + x_148
        x_154 = x_148 = None
        x_156 = torch.conv2d(
            x_155,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_157 = torch.nn.functional.batch_norm(
            x_156,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_156 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_158 = torch.nn.functional.silu(x_157, inplace=True)
        x_157 = None
        x_159 = torch.conv2d(
            x_158,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_158 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_160 = torch.nn.functional.batch_norm(
            x_159,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_159 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_161 = torch.nn.functional.silu(x_160, inplace=True)
        x_160 = None
        x_162 = x_161 + x_155
        x_161 = x_155 = None
        x_163 = torch.conv2d(
            x_162,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_164 = torch.nn.functional.batch_norm(
            x_163,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_163 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_165 = torch.nn.functional.silu(x_164, inplace=True)
        x_164 = None
        x_166 = torch.conv2d(
            x_165,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_165 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_167 = torch.nn.functional.batch_norm(
            x_166,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_166 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_168 = torch.nn.functional.silu(x_167, inplace=True)
        x_167 = None
        x_169 = x_168 + x_162
        x_168 = x_162 = None
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_171 = torch.nn.functional.batch_norm(
            x_170,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_170 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_172 = torch.nn.functional.silu(x_171, inplace=True)
        x_171 = None
        x_173 = torch.conv2d(
            x_172,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_172 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_174 = torch.nn.functional.batch_norm(
            x_173,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_173 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_175 = torch.nn.functional.silu(x_174, inplace=True)
        x_174 = None
        x_176 = x_175 + x_169
        x_175 = x_169 = None
        cat_2 = torch.cat([x_176, x2_2], dim=1)
        x_176 = x2_2 = None
        x_177 = torch.conv2d(
            cat_2,
            l_self_modules_stages_modules_2_modules_conv_transition_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_2 = l_self_modules_stages_modules_2_modules_conv_transition_modules_conv_parameters_weight_ = (None)
        x_178 = torch.nn.functional.batch_norm(
            x_177,
            l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_177 = l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_parameters_bias_ = (None)
        x_179 = torch.nn.functional.silu(x_178, inplace=True)
        x_178 = None
        x_180 = torch.conv2d(
            x_179,
            l_self_modules_stages_modules_3_modules_conv_down_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_179 = l_self_modules_stages_modules_3_modules_conv_down_modules_conv_parameters_weight_ = (None)
        x_181 = torch.nn.functional.batch_norm(
            x_180,
            l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_180 = l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_bias_ = (None)
        x_182 = torch.nn.functional.silu(x_181, inplace=True)
        x_181 = None
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_stages_modules_3_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_182 = l_self_modules_stages_modules_3_modules_conv_exp_modules_conv_parameters_weight_ = (None)
        x_184 = torch.nn.functional.batch_norm(
            x_183,
            l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_183 = l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_parameters_weight_ = (
            l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_parameters_bias_
        ) = None
        x_185 = torch.nn.functional.silu(x_184, inplace=True)
        x_184 = None
        split_3 = x_185.split(640, dim=1)
        x_185 = None
        x1_3 = split_3[0]
        x2_3 = split_3[1]
        split_3 = None
        x_186 = torch.conv2d(
            x1_3,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_187 = torch.nn.functional.batch_norm(
            x_186,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_186 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_188 = torch.nn.functional.silu(x_187, inplace=True)
        x_187 = None
        x_189 = torch.conv2d(
            x_188,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_188 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_190 = torch.nn.functional.batch_norm(
            x_189,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_189 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_191 = torch.nn.functional.silu(x_190, inplace=True)
        x_190 = None
        x_192 = x_191 + x1_3
        x_191 = x1_3 = None
        x_193 = torch.conv2d(
            x_192,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_194 = torch.nn.functional.batch_norm(
            x_193,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_193 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_195 = torch.nn.functional.silu(x_194, inplace=True)
        x_194 = None
        x_196 = torch.conv2d(
            x_195,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_195 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_197 = torch.nn.functional.batch_norm(
            x_196,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_196 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_198 = torch.nn.functional.silu(x_197, inplace=True)
        x_197 = None
        x_199 = x_198 + x_192
        x_198 = x_192 = None
        x_200 = torch.conv2d(
            x_199,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_201 = torch.nn.functional.batch_norm(
            x_200,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_200 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_202 = torch.nn.functional.silu(x_201, inplace=True)
        x_201 = None
        x_203 = torch.conv2d(
            x_202,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_202 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_204 = torch.nn.functional.batch_norm(
            x_203,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_203 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_205 = torch.nn.functional.silu(x_204, inplace=True)
        x_204 = None
        x_206 = x_205 + x_199
        x_205 = x_199 = None
        cat_3 = torch.cat([x_206, x2_3], dim=1)
        x_206 = x2_3 = None
        x_207 = torch.conv2d(
            cat_3,
            l_self_modules_stages_modules_3_modules_conv_transition_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_3 = l_self_modules_stages_modules_3_modules_conv_transition_modules_conv_parameters_weight_ = (None)
        x_208 = torch.nn.functional.batch_norm(
            x_207,
            l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_207 = l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_parameters_bias_ = (None)
        x_209 = torch.nn.functional.silu(x_208, inplace=True)
        x_208 = None
        x_210 = torch.nn.functional.adaptive_avg_pool2d(x_209, 1)
        x_209 = None
        x_211 = x_210.flatten(1, -1)
        x_210 = None
        x_212 = torch.nn.functional.dropout(x_211, 0.0, False, False)
        x_211 = None
        x_213 = torch._C._nn.linear(
            x_212,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_212 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_213,)
