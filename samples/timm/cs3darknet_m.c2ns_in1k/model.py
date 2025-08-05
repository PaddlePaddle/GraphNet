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
        split = x_11.split(48, dim=1)
        x_11 = None
        x1 = split[0]
        x2 = split[1]
        split = None
        x_12 = torch.conv2d(
            x1,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
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
            (1, 1),
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
            (0, 0),
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
            (1, 1),
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
        cat = torch.cat([x_25, x2], dim=1)
        x_25 = x2 = None
        x_26 = torch.conv2d(
            cat,
            l_self_modules_stages_modules_0_modules_conv_transition_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat = l_self_modules_stages_modules_0_modules_conv_transition_modules_conv_parameters_weight_ = (None)
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_parameters_bias_ = (None)
        x_28 = torch.nn.functional.silu(x_27, inplace=True)
        x_27 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_stages_modules_1_modules_conv_down_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_stages_modules_1_modules_conv_down_modules_conv_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_bias_ = (None)
        x_31 = torch.nn.functional.silu(x_30, inplace=True)
        x_30 = None
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_stages_modules_1_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_31 = l_self_modules_stages_modules_1_modules_conv_exp_modules_conv_parameters_weight_ = (None)
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_parameters_bias_
        ) = None
        x_34 = torch.nn.functional.silu(x_33, inplace=True)
        x_33 = None
        split_1 = x_34.split(96, dim=1)
        x_34 = None
        x1_1 = split_1[0]
        x2_1 = split_1[1]
        split_1 = None
        x_35 = torch.conv2d(
            x1_1,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_37 = torch.nn.functional.silu(x_36, inplace=True)
        x_36 = None
        x_38 = torch.conv2d(
            x_37,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_40 = torch.nn.functional.silu(x_39, inplace=True)
        x_39 = None
        x_41 = x_40 + x1_1
        x_40 = x1_1 = None
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_44 = torch.nn.functional.silu(x_43, inplace=True)
        x_43 = None
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_44 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_47 = torch.nn.functional.silu(x_46, inplace=True)
        x_46 = None
        x_48 = x_47 + x_41
        x_47 = x_41 = None
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_49 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_51 = torch.nn.functional.silu(x_50, inplace=True)
        x_50 = None
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_51 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_54 = torch.nn.functional.silu(x_53, inplace=True)
        x_53 = None
        x_55 = x_54 + x_48
        x_54 = x_48 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_58 = torch.nn.functional.silu(x_57, inplace=True)
        x_57 = None
        x_59 = torch.conv2d(
            x_58,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_58 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_59 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_61 = torch.nn.functional.silu(x_60, inplace=True)
        x_60 = None
        x_62 = x_61 + x_55
        x_61 = x_55 = None
        cat_1 = torch.cat([x_62, x2_1], dim=1)
        x_62 = x2_1 = None
        x_63 = torch.conv2d(
            cat_1,
            l_self_modules_stages_modules_1_modules_conv_transition_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_1 = l_self_modules_stages_modules_1_modules_conv_transition_modules_conv_parameters_weight_ = (None)
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_parameters_bias_ = (None)
        x_65 = torch.nn.functional.silu(x_64, inplace=True)
        x_64 = None
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_stages_modules_2_modules_conv_down_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_stages_modules_2_modules_conv_down_modules_conv_parameters_weight_ = (None)
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_bias_ = (None)
        x_68 = torch.nn.functional.silu(x_67, inplace=True)
        x_67 = None
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_stages_modules_2_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_68 = l_self_modules_stages_modules_2_modules_conv_exp_modules_conv_parameters_weight_ = (None)
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_parameters_bias_
        ) = None
        x_71 = torch.nn.functional.silu(x_70, inplace=True)
        x_70 = None
        split_2 = x_71.split(192, dim=1)
        x_71 = None
        x1_2 = split_2[0]
        x2_2 = split_2[1]
        split_2 = None
        x_72 = torch.conv2d(
            x1_2,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_74 = torch.nn.functional.silu(x_73, inplace=True)
        x_73 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_77 = torch.nn.functional.silu(x_76, inplace=True)
        x_76 = None
        x_78 = x_77 + x1_2
        x_77 = x1_2 = None
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_81 = torch.nn.functional.silu(x_80, inplace=True)
        x_80 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_81 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_84 = torch.nn.functional.silu(x_83, inplace=True)
        x_83 = None
        x_85 = x_84 + x_78
        x_84 = x_78 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_88 = torch.nn.functional.silu(x_87, inplace=True)
        x_87 = None
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_88 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_89 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_91 = torch.nn.functional.silu(x_90, inplace=True)
        x_90 = None
        x_92 = x_91 + x_85
        x_91 = x_85 = None
        x_93 = torch.conv2d(
            x_92,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_94 = torch.nn.functional.batch_norm(
            x_93,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_93 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_95 = torch.nn.functional.silu(x_94, inplace=True)
        x_94 = None
        x_96 = torch.conv2d(
            x_95,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_95 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_96 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_98 = torch.nn.functional.silu(x_97, inplace=True)
        x_97 = None
        x_99 = x_98 + x_92
        x_98 = x_92 = None
        x_100 = torch.conv2d(
            x_99,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_100 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_102 = torch.nn.functional.silu(x_101, inplace=True)
        x_101 = None
        x_103 = torch.conv2d(
            x_102,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_102 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_104 = torch.nn.functional.batch_norm(
            x_103,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_103 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_105 = torch.nn.functional.silu(x_104, inplace=True)
        x_104 = None
        x_106 = x_105 + x_99
        x_105 = x_99 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_109 = torch.nn.functional.silu(x_108, inplace=True)
        x_108 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_112 = torch.nn.functional.silu(x_111, inplace=True)
        x_111 = None
        x_113 = x_112 + x_106
        x_112 = x_106 = None
        cat_2 = torch.cat([x_113, x2_2], dim=1)
        x_113 = x2_2 = None
        x_114 = torch.conv2d(
            cat_2,
            l_self_modules_stages_modules_2_modules_conv_transition_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_2 = l_self_modules_stages_modules_2_modules_conv_transition_modules_conv_parameters_weight_ = (None)
        x_115 = torch.nn.functional.batch_norm(
            x_114,
            l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_114 = l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_parameters_bias_ = (None)
        x_116 = torch.nn.functional.silu(x_115, inplace=True)
        x_115 = None
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_stages_modules_3_modules_conv_down_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_116 = l_self_modules_stages_modules_3_modules_conv_down_modules_conv_parameters_weight_ = (None)
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_117 = l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_bias_ = (None)
        x_119 = torch.nn.functional.silu(x_118, inplace=True)
        x_118 = None
        x_120 = torch.conv2d(
            x_119,
            l_self_modules_stages_modules_3_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_119 = l_self_modules_stages_modules_3_modules_conv_exp_modules_conv_parameters_weight_ = (None)
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_parameters_weight_ = (
            l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_parameters_bias_
        ) = None
        x_122 = torch.nn.functional.silu(x_121, inplace=True)
        x_121 = None
        split_3 = x_122.split(384, dim=1)
        x_122 = None
        x1_3 = split_3[0]
        x2_3 = split_3[1]
        split_3 = None
        x_123 = torch.conv2d(
            x1_3,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_125 = torch.nn.functional.silu(x_124, inplace=True)
        x_124 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_128 = torch.nn.functional.silu(x_127, inplace=True)
        x_127 = None
        x_129 = x_128 + x1_3
        x_128 = x1_3 = None
        x_130 = torch.conv2d(
            x_129,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_130 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_132 = torch.nn.functional.silu(x_131, inplace=True)
        x_131 = None
        x_133 = torch.conv2d(
            x_132,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_132 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_134 = torch.nn.functional.batch_norm(
            x_133,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_133 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_135 = torch.nn.functional.silu(x_134, inplace=True)
        x_134 = None
        x_136 = x_135 + x_129
        x_135 = x_129 = None
        cat_3 = torch.cat([x_136, x2_3], dim=1)
        x_136 = x2_3 = None
        x_137 = torch.conv2d(
            cat_3,
            l_self_modules_stages_modules_3_modules_conv_transition_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_3 = l_self_modules_stages_modules_3_modules_conv_transition_modules_conv_parameters_weight_ = (None)
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_137 = l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_parameters_bias_ = (None)
        x_139 = torch.nn.functional.silu(x_138, inplace=True)
        x_138 = None
        x_140 = torch.nn.functional.adaptive_avg_pool2d(x_139, 1)
        x_139 = None
        x_141 = x_140.flatten(1, -1)
        x_140 = None
        x_142 = torch.nn.functional.dropout(x_141, 0.0, False, False)
        x_141 = None
        x_143 = torch._C._nn.linear(
            x_142,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_142 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_143,)
