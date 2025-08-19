import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_conv_transition_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_conv_transition_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_conv_transition_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_conv_transition_b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_0_modules_conv_transition_b_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_conv_transition_b_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_1_modules_conv_transition_b_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_conv_transition_b_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_2_modules_conv_transition_b_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_conv_transition_b_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_conv_transition_b_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_conv_transition_b_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_parameters_bias_
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
        split = x_11.split(128, dim=1)
        x_11 = None
        xs = split[0]
        xb = split[1]
        split = None
        x_12 = torch.conv2d(
            xb,
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
        x_se = x_17.mean((2, 3), keepdim=True)
        x_se_1 = torch.conv2d(
            x_se,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_2 = torch.nn.functional.silu(x_se_1, inplace=True)
        x_se_1 = None
        x_se_3 = torch.conv2d(
            x_se_2,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_2 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid = x_se_3.sigmoid()
        x_se_3 = None
        x_18 = x_17 * sigmoid
        x_17 = sigmoid = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_21 = x_20 + xb
        x_20 = xb = None
        x_22 = torch.nn.functional.silu(x_21, inplace=False)
        x_21 = None
        x_23 = torch.conv2d(
            x_22,
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
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_25 = torch.nn.functional.silu(x_24, inplace=True)
        x_24 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_28 = torch.nn.functional.silu(x_27, inplace=True)
        x_27 = None
        x_se_4 = x_28.mean((2, 3), keepdim=True)
        x_se_5 = torch.conv2d(
            x_se_4,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_4 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_6 = torch.nn.functional.silu(x_se_5, inplace=True)
        x_se_5 = None
        x_se_7 = torch.conv2d(
            x_se_6,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_6 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_1 = x_se_7.sigmoid()
        x_se_7 = None
        x_29 = x_28 * sigmoid_1
        x_28 = sigmoid_1 = None
        x_30 = torch.conv2d(
            x_29,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_29 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_30 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_32 = x_31 + x_22
        x_31 = x_22 = None
        x_33 = torch.nn.functional.silu(x_32, inplace=False)
        x_32 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_36 = torch.nn.functional.silu(x_35, inplace=True)
        x_35 = None
        x_37 = torch.conv2d(
            x_36,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_36 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_37 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_39 = torch.nn.functional.silu(x_38, inplace=True)
        x_38 = None
        x_se_8 = x_39.mean((2, 3), keepdim=True)
        x_se_9 = torch.conv2d(
            x_se_8,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_8 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_10 = torch.nn.functional.silu(x_se_9, inplace=True)
        x_se_9 = None
        x_se_11 = torch.conv2d(
            x_se_10,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_10 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_2 = x_se_11.sigmoid()
        x_se_11 = None
        x_40 = x_39 * sigmoid_2
        x_39 = sigmoid_2 = None
        x_41 = torch.conv2d(
            x_40,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_40 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_42 = torch.nn.functional.batch_norm(
            x_41,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_41 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_43 = x_42 + x_33
        x_42 = x_33 = None
        x_44 = torch.nn.functional.silu(x_43, inplace=False)
        x_43 = None
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_stages_modules_0_modules_conv_transition_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_44 = l_self_modules_stages_modules_0_modules_conv_transition_b_modules_conv_parameters_weight_ = (None)
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = l_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_conv_transition_b_modules_bn_parameters_bias_ = (None)
        x_47 = torch.nn.functional.silu(x_46, inplace=True)
        x_46 = None
        xb_1 = x_47.contiguous()
        x_47 = None
        cat = torch.cat([xs, xb_1], dim=1)
        xs = xb_1 = None
        x_48 = torch.conv2d(
            cat,
            l_self_modules_stages_modules_0_modules_conv_transition_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat = l_self_modules_stages_modules_0_modules_conv_transition_modules_conv_parameters_weight_ = (None)
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_48 = l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_conv_transition_modules_bn_parameters_bias_ = (None)
        x_50 = torch.nn.functional.silu(x_49, inplace=True)
        x_49 = None
        x_51 = torch.conv2d(
            x_50,
            l_self_modules_stages_modules_1_modules_conv_down_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_50 = l_self_modules_stages_modules_1_modules_conv_down_modules_conv_parameters_weight_ = (None)
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_51 = l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_conv_down_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_conv_down_modules_bn_parameters_bias_ = (None)
        x_53 = torch.nn.functional.silu(x_52, inplace=True)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_stages_modules_1_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_stages_modules_1_modules_conv_exp_modules_conv_parameters_weight_ = (None)
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_parameters_weight_ = (
            l_self_modules_stages_modules_1_modules_conv_exp_modules_bn_parameters_bias_
        ) = None
        x_56 = torch.nn.functional.silu(x_55, inplace=True)
        x_55 = None
        split_1 = x_56.split(256, dim=1)
        x_56 = None
        xs_1 = split_1[0]
        xb_2 = split_1[1]
        split_1 = None
        x_57 = torch.conv2d(
            xb_2,
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
        x_58 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_57 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_59 = torch.nn.functional.silu(x_58, inplace=True)
        x_58 = None
        x_60 = torch.conv2d(
            x_59,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_59 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_60 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_62 = torch.nn.functional.silu(x_61, inplace=True)
        x_61 = None
        x_se_12 = x_62.mean((2, 3), keepdim=True)
        x_se_13 = torch.conv2d(
            x_se_12,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_12 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_14 = torch.nn.functional.silu(x_se_13, inplace=True)
        x_se_13 = None
        x_se_15 = torch.conv2d(
            x_se_14,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_14 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_3 = x_se_15.sigmoid()
        x_se_15 = None
        x_63 = x_62 * sigmoid_3
        x_62 = sigmoid_3 = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_66 = x_65 + xb_2
        x_65 = xb_2 = None
        x_67 = torch.nn.functional.silu(x_66, inplace=False)
        x_66 = None
        x_68 = torch.conv2d(
            x_67,
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
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_70 = torch.nn.functional.silu(x_69, inplace=True)
        x_69 = None
        x_71 = torch.conv2d(
            x_70,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_70 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_72 = torch.nn.functional.batch_norm(
            x_71,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_71 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_73 = torch.nn.functional.silu(x_72, inplace=True)
        x_72 = None
        x_se_16 = x_73.mean((2, 3), keepdim=True)
        x_se_17 = torch.conv2d(
            x_se_16,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_16 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_18 = torch.nn.functional.silu(x_se_17, inplace=True)
        x_se_17 = None
        x_se_19 = torch.conv2d(
            x_se_18,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_18 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_4 = x_se_19.sigmoid()
        x_se_19 = None
        x_74 = x_73 * sigmoid_4
        x_73 = sigmoid_4 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_77 = x_76 + x_67
        x_76 = x_67 = None
        x_78 = torch.nn.functional.silu(x_77, inplace=False)
        x_77 = None
        x_79 = torch.conv2d(
            x_78,
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
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_81 = torch.nn.functional.silu(x_80, inplace=True)
        x_80 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_81 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_84 = torch.nn.functional.silu(x_83, inplace=True)
        x_83 = None
        x_se_20 = x_84.mean((2, 3), keepdim=True)
        x_se_21 = torch.conv2d(
            x_se_20,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_20 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_22 = torch.nn.functional.silu(x_se_21, inplace=True)
        x_se_21 = None
        x_se_23 = torch.conv2d(
            x_se_22,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_22 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_5 = x_se_23.sigmoid()
        x_se_23 = None
        x_85 = x_84 * sigmoid_5
        x_84 = sigmoid_5 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_85 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_88 = x_87 + x_78
        x_87 = x_78 = None
        x_89 = torch.nn.functional.silu(x_88, inplace=False)
        x_88 = None
        x_90 = torch.conv2d(
            x_89,
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
        x_91 = torch.nn.functional.batch_norm(
            x_90,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_90 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_92 = torch.nn.functional.silu(x_91, inplace=True)
        x_91 = None
        x_93 = torch.conv2d(
            x_92,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_92 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_94 = torch.nn.functional.batch_norm(
            x_93,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_93 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_95 = torch.nn.functional.silu(x_94, inplace=True)
        x_94 = None
        x_se_24 = x_95.mean((2, 3), keepdim=True)
        x_se_25 = torch.conv2d(
            x_se_24,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_24 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_26 = torch.nn.functional.silu(x_se_25, inplace=True)
        x_se_25 = None
        x_se_27 = torch.conv2d(
            x_se_26,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_26 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_6 = x_se_27.sigmoid()
        x_se_27 = None
        x_96 = x_95 * sigmoid_6
        x_95 = sigmoid_6 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_96 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_97 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_99 = x_98 + x_89
        x_98 = x_89 = None
        x_100 = torch.nn.functional.silu(x_99, inplace=False)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_103 = torch.nn.functional.silu(x_102, inplace=True)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_104 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_106 = torch.nn.functional.silu(x_105, inplace=True)
        x_105 = None
        x_se_28 = x_106.mean((2, 3), keepdim=True)
        x_se_29 = torch.conv2d(
            x_se_28,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_28 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_30 = torch.nn.functional.silu(x_se_29, inplace=True)
        x_se_29 = None
        x_se_31 = torch.conv2d(
            x_se_30,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_30 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_7 = x_se_31.sigmoid()
        x_se_31 = None
        x_107 = x_106 * sigmoid_7
        x_106 = sigmoid_7 = None
        x_108 = torch.conv2d(
            x_107,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_107 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_108 = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_110 = x_109 + x_100
        x_109 = x_100 = None
        x_111 = torch.nn.functional.silu(x_110, inplace=False)
        x_110 = None
        x_112 = torch.conv2d(
            x_111,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_112 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_114 = torch.nn.functional.silu(x_113, inplace=True)
        x_113 = None
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_114 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_115 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_117 = torch.nn.functional.silu(x_116, inplace=True)
        x_116 = None
        x_se_32 = x_117.mean((2, 3), keepdim=True)
        x_se_33 = torch.conv2d(
            x_se_32,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_32 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_34 = torch.nn.functional.silu(x_se_33, inplace=True)
        x_se_33 = None
        x_se_35 = torch.conv2d(
            x_se_34,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_34 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_8 = x_se_35.sigmoid()
        x_se_35 = None
        x_118 = x_117 * sigmoid_8
        x_117 = sigmoid_8 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_121 = x_120 + x_111
        x_120 = x_111 = None
        x_122 = torch.nn.functional.silu(x_121, inplace=False)
        x_121 = None
        x_123 = torch.conv2d(
            x_122,
            l_self_modules_stages_modules_1_modules_conv_transition_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_122 = l_self_modules_stages_modules_1_modules_conv_transition_b_modules_conv_parameters_weight_ = (None)
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = l_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_conv_transition_b_modules_bn_parameters_bias_ = (None)
        x_125 = torch.nn.functional.silu(x_124, inplace=True)
        x_124 = None
        xb_3 = x_125.contiguous()
        x_125 = None
        cat_1 = torch.cat([xs_1, xb_3], dim=1)
        xs_1 = xb_3 = None
        x_126 = torch.conv2d(
            cat_1,
            l_self_modules_stages_modules_1_modules_conv_transition_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_1 = l_self_modules_stages_modules_1_modules_conv_transition_modules_conv_parameters_weight_ = (None)
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_conv_transition_modules_bn_parameters_bias_ = (None)
        x_128 = torch.nn.functional.silu(x_127, inplace=True)
        x_127 = None
        x_129 = torch.conv2d(
            x_128,
            l_self_modules_stages_modules_2_modules_conv_down_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            256,
        )
        x_128 = l_self_modules_stages_modules_2_modules_conv_down_modules_conv_parameters_weight_ = (None)
        x_130 = torch.nn.functional.batch_norm(
            x_129,
            l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_129 = l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_conv_down_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_conv_down_modules_bn_parameters_bias_ = (None)
        x_131 = torch.nn.functional.silu(x_130, inplace=True)
        x_130 = None
        x_132 = torch.conv2d(
            x_131,
            l_self_modules_stages_modules_2_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_131 = l_self_modules_stages_modules_2_modules_conv_exp_modules_conv_parameters_weight_ = (None)
        x_133 = torch.nn.functional.batch_norm(
            x_132,
            l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_132 = l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_parameters_weight_ = (
            l_self_modules_stages_modules_2_modules_conv_exp_modules_bn_parameters_bias_
        ) = None
        x_134 = torch.nn.functional.silu(x_133, inplace=True)
        x_133 = None
        split_2 = x_134.split(512, dim=1)
        x_134 = None
        xs_2 = split_2[0]
        xb_4 = split_2[1]
        split_2 = None
        x_135 = torch.conv2d(
            xb_4,
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
        x_136 = torch.nn.functional.batch_norm(
            x_135,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_135 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_137 = torch.nn.functional.silu(x_136, inplace=True)
        x_136 = None
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_137 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_139 = torch.nn.functional.batch_norm(
            x_138,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_138 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_140 = torch.nn.functional.silu(x_139, inplace=True)
        x_139 = None
        x_se_36 = x_140.mean((2, 3), keepdim=True)
        x_se_37 = torch.conv2d(
            x_se_36,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_36 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_38 = torch.nn.functional.silu(x_se_37, inplace=True)
        x_se_37 = None
        x_se_39 = torch.conv2d(
            x_se_38,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_38 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_9 = x_se_39.sigmoid()
        x_se_39 = None
        x_141 = x_140 * sigmoid_9
        x_140 = sigmoid_9 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_141 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_142 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_144 = x_143 + xb_4
        x_143 = xb_4 = None
        x_145 = torch.nn.functional.silu(x_144, inplace=False)
        x_144 = None
        x_146 = torch.conv2d(
            x_145,
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
        x_147 = torch.nn.functional.batch_norm(
            x_146,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_146 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_148 = torch.nn.functional.silu(x_147, inplace=True)
        x_147 = None
        x_149 = torch.conv2d(
            x_148,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_148 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_150 = torch.nn.functional.batch_norm(
            x_149,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_149 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_151 = torch.nn.functional.silu(x_150, inplace=True)
        x_150 = None
        x_se_40 = x_151.mean((2, 3), keepdim=True)
        x_se_41 = torch.conv2d(
            x_se_40,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_40 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_42 = torch.nn.functional.silu(x_se_41, inplace=True)
        x_se_41 = None
        x_se_43 = torch.conv2d(
            x_se_42,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_42 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_10 = x_se_43.sigmoid()
        x_se_43 = None
        x_152 = x_151 * sigmoid_10
        x_151 = sigmoid_10 = None
        x_153 = torch.conv2d(
            x_152,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_152 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_154 = torch.nn.functional.batch_norm(
            x_153,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_153 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_155 = x_154 + x_145
        x_154 = x_145 = None
        x_156 = torch.nn.functional.silu(x_155, inplace=False)
        x_155 = None
        x_157 = torch.conv2d(
            x_156,
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
        x_158 = torch.nn.functional.batch_norm(
            x_157,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_157 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_159 = torch.nn.functional.silu(x_158, inplace=True)
        x_158 = None
        x_160 = torch.conv2d(
            x_159,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_159 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_161 = torch.nn.functional.batch_norm(
            x_160,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_160 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_162 = torch.nn.functional.silu(x_161, inplace=True)
        x_161 = None
        x_se_44 = x_162.mean((2, 3), keepdim=True)
        x_se_45 = torch.conv2d(
            x_se_44,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_44 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_46 = torch.nn.functional.silu(x_se_45, inplace=True)
        x_se_45 = None
        x_se_47 = torch.conv2d(
            x_se_46,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_46 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_11 = x_se_47.sigmoid()
        x_se_47 = None
        x_163 = x_162 * sigmoid_11
        x_162 = sigmoid_11 = None
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_163 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_165 = torch.nn.functional.batch_norm(
            x_164,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_164 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_166 = x_165 + x_156
        x_165 = x_156 = None
        x_167 = torch.nn.functional.silu(x_166, inplace=False)
        x_166 = None
        x_168 = torch.conv2d(
            x_167,
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
        x_169 = torch.nn.functional.batch_norm(
            x_168,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_168 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_170 = torch.nn.functional.silu(x_169, inplace=True)
        x_169 = None
        x_171 = torch.conv2d(
            x_170,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_170 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_172 = torch.nn.functional.batch_norm(
            x_171,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_171 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_173 = torch.nn.functional.silu(x_172, inplace=True)
        x_172 = None
        x_se_48 = x_173.mean((2, 3), keepdim=True)
        x_se_49 = torch.conv2d(
            x_se_48,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_48 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_50 = torch.nn.functional.silu(x_se_49, inplace=True)
        x_se_49 = None
        x_se_51 = torch.conv2d(
            x_se_50,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_50 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_12 = x_se_51.sigmoid()
        x_se_51 = None
        x_174 = x_173 * sigmoid_12
        x_173 = sigmoid_12 = None
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_174 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_176 = torch.nn.functional.batch_norm(
            x_175,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_175 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_177 = x_176 + x_167
        x_176 = x_167 = None
        x_178 = torch.nn.functional.silu(x_177, inplace=False)
        x_177 = None
        x_179 = torch.conv2d(
            x_178,
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
        x_180 = torch.nn.functional.batch_norm(
            x_179,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_179 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_181 = torch.nn.functional.silu(x_180, inplace=True)
        x_180 = None
        x_182 = torch.conv2d(
            x_181,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_181 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_183 = torch.nn.functional.batch_norm(
            x_182,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_182 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_184 = torch.nn.functional.silu(x_183, inplace=True)
        x_183 = None
        x_se_52 = x_184.mean((2, 3), keepdim=True)
        x_se_53 = torch.conv2d(
            x_se_52,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_52 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_54 = torch.nn.functional.silu(x_se_53, inplace=True)
        x_se_53 = None
        x_se_55 = torch.conv2d(
            x_se_54,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_54 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_13 = x_se_55.sigmoid()
        x_se_55 = None
        x_185 = x_184 * sigmoid_13
        x_184 = sigmoid_13 = None
        x_186 = torch.conv2d(
            x_185,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_185 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_187 = torch.nn.functional.batch_norm(
            x_186,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_186 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_188 = x_187 + x_178
        x_187 = x_178 = None
        x_189 = torch.nn.functional.silu(x_188, inplace=False)
        x_188 = None
        x_190 = torch.conv2d(
            x_189,
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
        x_191 = torch.nn.functional.batch_norm(
            x_190,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_190 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_192 = torch.nn.functional.silu(x_191, inplace=True)
        x_191 = None
        x_193 = torch.conv2d(
            x_192,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_192 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_194 = torch.nn.functional.batch_norm(
            x_193,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_193 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_195 = torch.nn.functional.silu(x_194, inplace=True)
        x_194 = None
        x_se_56 = x_195.mean((2, 3), keepdim=True)
        x_se_57 = torch.conv2d(
            x_se_56,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_56 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_58 = torch.nn.functional.silu(x_se_57, inplace=True)
        x_se_57 = None
        x_se_59 = torch.conv2d(
            x_se_58,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_58 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_14 = x_se_59.sigmoid()
        x_se_59 = None
        x_196 = x_195 * sigmoid_14
        x_195 = sigmoid_14 = None
        x_197 = torch.conv2d(
            x_196,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_196 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_198 = torch.nn.functional.batch_norm(
            x_197,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_197 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_199 = x_198 + x_189
        x_198 = x_189 = None
        x_200 = torch.nn.functional.silu(x_199, inplace=False)
        x_199 = None
        x_201 = torch.conv2d(
            x_200,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_202 = torch.nn.functional.batch_norm(
            x_201,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_201 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_203 = torch.nn.functional.silu(x_202, inplace=True)
        x_202 = None
        x_204 = torch.conv2d(
            x_203,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_203 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_205 = torch.nn.functional.batch_norm(
            x_204,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_204 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_206 = torch.nn.functional.silu(x_205, inplace=True)
        x_205 = None
        x_se_60 = x_206.mean((2, 3), keepdim=True)
        x_se_61 = torch.conv2d(
            x_se_60,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_60 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_62 = torch.nn.functional.silu(x_se_61, inplace=True)
        x_se_61 = None
        x_se_63 = torch.conv2d(
            x_se_62,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_62 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_15 = x_se_63.sigmoid()
        x_se_63 = None
        x_207 = x_206 * sigmoid_15
        x_206 = sigmoid_15 = None
        x_208 = torch.conv2d(
            x_207,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_207 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_209 = torch.nn.functional.batch_norm(
            x_208,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_208 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_210 = x_209 + x_200
        x_209 = x_200 = None
        x_211 = torch.nn.functional.silu(x_210, inplace=False)
        x_210 = None
        x_212 = torch.conv2d(
            x_211,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_213 = torch.nn.functional.batch_norm(
            x_212,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_212 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_214 = torch.nn.functional.silu(x_213, inplace=True)
        x_213 = None
        x_215 = torch.conv2d(
            x_214,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_214 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_216 = torch.nn.functional.batch_norm(
            x_215,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_215 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_217 = torch.nn.functional.silu(x_216, inplace=True)
        x_216 = None
        x_se_64 = x_217.mean((2, 3), keepdim=True)
        x_se_65 = torch.conv2d(
            x_se_64,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_64 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_66 = torch.nn.functional.silu(x_se_65, inplace=True)
        x_se_65 = None
        x_se_67 = torch.conv2d(
            x_se_66,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_66 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_16 = x_se_67.sigmoid()
        x_se_67 = None
        x_218 = x_217 * sigmoid_16
        x_217 = sigmoid_16 = None
        x_219 = torch.conv2d(
            x_218,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_218 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_220 = torch.nn.functional.batch_norm(
            x_219,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_219 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_221 = x_220 + x_211
        x_220 = x_211 = None
        x_222 = torch.nn.functional.silu(x_221, inplace=False)
        x_221 = None
        x_223 = torch.conv2d(
            x_222,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_224 = torch.nn.functional.batch_norm(
            x_223,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_223 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_225 = torch.nn.functional.silu(x_224, inplace=True)
        x_224 = None
        x_226 = torch.conv2d(
            x_225,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_225 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_227 = torch.nn.functional.batch_norm(
            x_226,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_226 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_228 = torch.nn.functional.silu(x_227, inplace=True)
        x_227 = None
        x_se_68 = x_228.mean((2, 3), keepdim=True)
        x_se_69 = torch.conv2d(
            x_se_68,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_68 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_70 = torch.nn.functional.silu(x_se_69, inplace=True)
        x_se_69 = None
        x_se_71 = torch.conv2d(
            x_se_70,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_70 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_17 = x_se_71.sigmoid()
        x_se_71 = None
        x_229 = x_228 * sigmoid_17
        x_228 = sigmoid_17 = None
        x_230 = torch.conv2d(
            x_229,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_229 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_231 = torch.nn.functional.batch_norm(
            x_230,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_230 = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_232 = x_231 + x_222
        x_231 = x_222 = None
        x_233 = torch.nn.functional.silu(x_232, inplace=False)
        x_232 = None
        x_234 = torch.conv2d(
            x_233,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_235 = torch.nn.functional.batch_norm(
            x_234,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_234 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_236 = torch.nn.functional.silu(x_235, inplace=True)
        x_235 = None
        x_237 = torch.conv2d(
            x_236,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_236 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_238 = torch.nn.functional.batch_norm(
            x_237,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_237 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_239 = torch.nn.functional.silu(x_238, inplace=True)
        x_238 = None
        x_se_72 = x_239.mean((2, 3), keepdim=True)
        x_se_73 = torch.conv2d(
            x_se_72,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_72 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_74 = torch.nn.functional.silu(x_se_73, inplace=True)
        x_se_73 = None
        x_se_75 = torch.conv2d(
            x_se_74,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_74 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_18 = x_se_75.sigmoid()
        x_se_75 = None
        x_240 = x_239 * sigmoid_18
        x_239 = sigmoid_18 = None
        x_241 = torch.conv2d(
            x_240,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_240 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_242 = torch.nn.functional.batch_norm(
            x_241,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_241 = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_243 = x_242 + x_233
        x_242 = x_233 = None
        x_244 = torch.nn.functional.silu(x_243, inplace=False)
        x_243 = None
        x_245 = torch.conv2d(
            x_244,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_246 = torch.nn.functional.batch_norm(
            x_245,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_245 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_247 = torch.nn.functional.silu(x_246, inplace=True)
        x_246 = None
        x_248 = torch.conv2d(
            x_247,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_247 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_249 = torch.nn.functional.batch_norm(
            x_248,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_248 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_250 = torch.nn.functional.silu(x_249, inplace=True)
        x_249 = None
        x_se_76 = x_250.mean((2, 3), keepdim=True)
        x_se_77 = torch.conv2d(
            x_se_76,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_76 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_78 = torch.nn.functional.silu(x_se_77, inplace=True)
        x_se_77 = None
        x_se_79 = torch.conv2d(
            x_se_78,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_78 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_19 = x_se_79.sigmoid()
        x_se_79 = None
        x_251 = x_250 * sigmoid_19
        x_250 = sigmoid_19 = None
        x_252 = torch.conv2d(
            x_251,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_251 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_253 = torch.nn.functional.batch_norm(
            x_252,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_252 = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_254 = x_253 + x_244
        x_253 = x_244 = None
        x_255 = torch.nn.functional.silu(x_254, inplace=False)
        x_254 = None
        x_256 = torch.conv2d(
            x_255,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_257 = torch.nn.functional.batch_norm(
            x_256,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_256 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_258 = torch.nn.functional.silu(x_257, inplace=True)
        x_257 = None
        x_259 = torch.conv2d(
            x_258,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_258 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_260 = torch.nn.functional.batch_norm(
            x_259,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_259 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_261 = torch.nn.functional.silu(x_260, inplace=True)
        x_260 = None
        x_se_80 = x_261.mean((2, 3), keepdim=True)
        x_se_81 = torch.conv2d(
            x_se_80,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_80 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_82 = torch.nn.functional.silu(x_se_81, inplace=True)
        x_se_81 = None
        x_se_83 = torch.conv2d(
            x_se_82,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_82 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_20 = x_se_83.sigmoid()
        x_se_83 = None
        x_262 = x_261 * sigmoid_20
        x_261 = sigmoid_20 = None
        x_263 = torch.conv2d(
            x_262,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_262 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_264 = torch.nn.functional.batch_norm(
            x_263,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_263 = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_265 = x_264 + x_255
        x_264 = x_255 = None
        x_266 = torch.nn.functional.silu(x_265, inplace=False)
        x_265 = None
        x_267 = torch.conv2d(
            x_266,
            l_self_modules_stages_modules_2_modules_conv_transition_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_266 = l_self_modules_stages_modules_2_modules_conv_transition_b_modules_conv_parameters_weight_ = (None)
        x_268 = torch.nn.functional.batch_norm(
            x_267,
            l_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_267 = l_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_conv_transition_b_modules_bn_parameters_bias_ = (None)
        x_269 = torch.nn.functional.silu(x_268, inplace=True)
        x_268 = None
        xb_5 = x_269.contiguous()
        x_269 = None
        cat_2 = torch.cat([xs_2, xb_5], dim=1)
        xs_2 = xb_5 = None
        x_270 = torch.conv2d(
            cat_2,
            l_self_modules_stages_modules_2_modules_conv_transition_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_2 = l_self_modules_stages_modules_2_modules_conv_transition_modules_conv_parameters_weight_ = (None)
        x_271 = torch.nn.functional.batch_norm(
            x_270,
            l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_270 = l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_conv_transition_modules_bn_parameters_bias_ = (None)
        x_272 = torch.nn.functional.silu(x_271, inplace=True)
        x_271 = None
        x_273 = torch.conv2d(
            x_272,
            l_self_modules_stages_modules_3_modules_conv_down_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            512,
        )
        x_272 = l_self_modules_stages_modules_3_modules_conv_down_modules_conv_parameters_weight_ = (None)
        x_274 = torch.nn.functional.batch_norm(
            x_273,
            l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_273 = l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_conv_down_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_conv_down_modules_bn_parameters_bias_ = (None)
        x_275 = torch.nn.functional.silu(x_274, inplace=True)
        x_274 = None
        x_276 = torch.conv2d(
            x_275,
            l_self_modules_stages_modules_3_modules_conv_exp_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_275 = l_self_modules_stages_modules_3_modules_conv_exp_modules_conv_parameters_weight_ = (None)
        x_277 = torch.nn.functional.batch_norm(
            x_276,
            l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_276 = l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_parameters_weight_ = (
            l_self_modules_stages_modules_3_modules_conv_exp_modules_bn_parameters_bias_
        ) = None
        x_278 = torch.nn.functional.silu(x_277, inplace=True)
        x_277 = None
        split_3 = x_278.split(1024, dim=1)
        x_278 = None
        xs_3 = split_3[0]
        xb_6 = split_3[1]
        split_3 = None
        x_279 = torch.conv2d(
            xb_6,
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
        x_280 = torch.nn.functional.batch_norm(
            x_279,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_279 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_281 = torch.nn.functional.silu(x_280, inplace=True)
        x_280 = None
        x_282 = torch.conv2d(
            x_281,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_281 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_283 = torch.nn.functional.batch_norm(
            x_282,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_282 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_284 = torch.nn.functional.silu(x_283, inplace=True)
        x_283 = None
        x_se_84 = x_284.mean((2, 3), keepdim=True)
        x_se_85 = torch.conv2d(
            x_se_84,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_84 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_86 = torch.nn.functional.silu(x_se_85, inplace=True)
        x_se_85 = None
        x_se_87 = torch.conv2d(
            x_se_86,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_86 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_21 = x_se_87.sigmoid()
        x_se_87 = None
        x_285 = x_284 * sigmoid_21
        x_284 = sigmoid_21 = None
        x_286 = torch.conv2d(
            x_285,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_285 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_287 = torch.nn.functional.batch_norm(
            x_286,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_286 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_288 = x_287 + xb_6
        x_287 = xb_6 = None
        x_289 = torch.nn.functional.silu(x_288, inplace=False)
        x_288 = None
        x_290 = torch.conv2d(
            x_289,
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
        x_291 = torch.nn.functional.batch_norm(
            x_290,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_290 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_292 = torch.nn.functional.silu(x_291, inplace=True)
        x_291 = None
        x_293 = torch.conv2d(
            x_292,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_292 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_294 = torch.nn.functional.batch_norm(
            x_293,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_293 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_295 = torch.nn.functional.silu(x_294, inplace=True)
        x_294 = None
        x_se_88 = x_295.mean((2, 3), keepdim=True)
        x_se_89 = torch.conv2d(
            x_se_88,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_88 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_90 = torch.nn.functional.silu(x_se_89, inplace=True)
        x_se_89 = None
        x_se_91 = torch.conv2d(
            x_se_90,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_90 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_22 = x_se_91.sigmoid()
        x_se_91 = None
        x_296 = x_295 * sigmoid_22
        x_295 = sigmoid_22 = None
        x_297 = torch.conv2d(
            x_296,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_296 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_298 = torch.nn.functional.batch_norm(
            x_297,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_297 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_299 = x_298 + x_289
        x_298 = x_289 = None
        x_300 = torch.nn.functional.silu(x_299, inplace=False)
        x_299 = None
        x_301 = torch.conv2d(
            x_300,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_302 = torch.nn.functional.batch_norm(
            x_301,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_301 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_303 = torch.nn.functional.silu(x_302, inplace=True)
        x_302 = None
        x_304 = torch.conv2d(
            x_303,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_303 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_305 = torch.nn.functional.batch_norm(
            x_304,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_304 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_306 = torch.nn.functional.silu(x_305, inplace=True)
        x_305 = None
        x_se_92 = x_306.mean((2, 3), keepdim=True)
        x_se_93 = torch.conv2d(
            x_se_92,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_92 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_94 = torch.nn.functional.silu(x_se_93, inplace=True)
        x_se_93 = None
        x_se_95 = torch.conv2d(
            x_se_94,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_94 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_23 = x_se_95.sigmoid()
        x_se_95 = None
        x_307 = x_306 * sigmoid_23
        x_306 = sigmoid_23 = None
        x_308 = torch.conv2d(
            x_307,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_307 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_309 = torch.nn.functional.batch_norm(
            x_308,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_308 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_310 = x_309 + x_300
        x_309 = x_300 = None
        x_311 = torch.nn.functional.silu(x_310, inplace=False)
        x_310 = None
        x_312 = torch.conv2d(
            x_311,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_313 = torch.nn.functional.batch_norm(
            x_312,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_312 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_314 = torch.nn.functional.silu(x_313, inplace=True)
        x_313 = None
        x_315 = torch.conv2d(
            x_314,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            512,
        )
        x_314 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_316 = torch.nn.functional.batch_norm(
            x_315,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_315 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_317 = torch.nn.functional.silu(x_316, inplace=True)
        x_316 = None
        x_se_96 = x_317.mean((2, 3), keepdim=True)
        x_se_97 = torch.conv2d(
            x_se_96,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_96 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc1_parameters_bias_ = (None)
        x_se_98 = torch.nn.functional.silu(x_se_97, inplace=True)
        x_se_97 = None
        x_se_99 = torch.conv2d(
            x_se_98,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_98 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_attn2_modules_fc2_parameters_bias_ = (None)
        sigmoid_24 = x_se_99.sigmoid()
        x_se_99 = None
        x_318 = x_317 * sigmoid_24
        x_317 = sigmoid_24 = None
        x_319 = torch.conv2d(
            x_318,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_318 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_320 = torch.nn.functional.batch_norm(
            x_319,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_319 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv3_modules_bn_parameters_bias_ = (None)
        x_321 = x_320 + x_311
        x_320 = x_311 = None
        x_322 = torch.nn.functional.silu(x_321, inplace=False)
        x_321 = None
        x_323 = torch.conv2d(
            x_322,
            l_self_modules_stages_modules_3_modules_conv_transition_b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_322 = l_self_modules_stages_modules_3_modules_conv_transition_b_modules_conv_parameters_weight_ = (None)
        x_324 = torch.nn.functional.batch_norm(
            x_323,
            l_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_323 = l_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_conv_transition_b_modules_bn_parameters_bias_ = (None)
        x_325 = torch.nn.functional.silu(x_324, inplace=True)
        x_324 = None
        xb_7 = x_325.contiguous()
        x_325 = None
        cat_3 = torch.cat([xs_3, xb_7], dim=1)
        xs_3 = xb_7 = None
        x_326 = torch.conv2d(
            cat_3,
            l_self_modules_stages_modules_3_modules_conv_transition_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        cat_3 = l_self_modules_stages_modules_3_modules_conv_transition_modules_conv_parameters_weight_ = (None)
        x_327 = torch.nn.functional.batch_norm(
            x_326,
            l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_326 = l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_conv_transition_modules_bn_parameters_bias_ = (None)
        x_328 = torch.nn.functional.silu(x_327, inplace=True)
        x_327 = None
        x_329 = torch.nn.functional.adaptive_avg_pool2d(x_328, 1)
        x_328 = None
        x_330 = x_329.flatten(1, -1)
        x_329 = None
        x_331 = torch.nn.functional.dropout(x_330, 0.0, False, False)
        x_330 = None
        x_332 = torch._C._nn.linear(
            x_331,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_331 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_332,)
