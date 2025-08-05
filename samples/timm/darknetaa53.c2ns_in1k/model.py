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
        L_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_parameters_bias_
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
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_
        l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_ = L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_
        l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_ = L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_
        l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_ = L_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_
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
            (1, 1),
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
        x_2 = torch.nn.functional.leaky_relu(x_1, 0.01, True)
        x_1 = None
        input_1 = torch._C._nn.avg_pool2d(x_2, 2, 2, 0, False, True, None)
        x_2 = None
        x_3 = torch.conv2d(
            input_1,
            l_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_1 = l_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_conv_parameters_weight_ = (None)
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = l_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_conv_down_modules_1_modules_bn_parameters_bias_ = (None)
        x_5 = torch.nn.functional.leaky_relu(x_4, 0.01, True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
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
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_8 = torch.nn.functional.leaky_relu(x_7, 0.01, True)
        x_7 = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_8 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_11 = torch.nn.functional.leaky_relu(x_10, 0.01, True)
        x_10 = None
        x_12 = x_11 + x_5
        x_11 = x_5 = None
        input_2 = torch._C._nn.avg_pool2d(x_12, 2, 2, 0, False, True, None)
        x_12 = None
        x_13 = torch.conv2d(
            input_2,
            l_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_2 = l_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_conv_parameters_weight_ = (None)
        x_14 = torch.nn.functional.batch_norm(
            x_13,
            l_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_13 = l_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_conv_down_modules_1_modules_bn_parameters_bias_ = (None)
        x_15 = torch.nn.functional.leaky_relu(x_14, 0.01, True)
        x_14 = None
        x_16 = torch.conv2d(
            x_15,
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
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_18 = torch.nn.functional.leaky_relu(x_17, 0.01, True)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_21 = torch.nn.functional.leaky_relu(x_20, 0.01, True)
        x_20 = None
        x_22 = x_21 + x_15
        x_21 = x_15 = None
        x_23 = torch.conv2d(
            x_22,
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
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_25 = torch.nn.functional.leaky_relu(x_24, 0.01, True)
        x_24 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_26 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_28 = torch.nn.functional.leaky_relu(x_27, 0.01, True)
        x_27 = None
        x_29 = x_28 + x_22
        x_28 = x_22 = None
        input_3 = torch._C._nn.avg_pool2d(x_29, 2, 2, 0, False, True, None)
        x_29 = None
        x_30 = torch.conv2d(
            input_3,
            l_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_3 = l_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_conv_parameters_weight_ = (None)
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_30 = l_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_conv_down_modules_1_modules_bn_parameters_bias_ = (None)
        x_32 = torch.nn.functional.leaky_relu(x_31, 0.01, True)
        x_31 = None
        x_33 = torch.conv2d(
            x_32,
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
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_35 = torch.nn.functional.leaky_relu(x_34, 0.01, True)
        x_34 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_38 = torch.nn.functional.leaky_relu(x_37, 0.01, True)
        x_37 = None
        x_39 = x_38 + x_32
        x_38 = x_32 = None
        x_40 = torch.conv2d(
            x_39,
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
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_42 = torch.nn.functional.leaky_relu(x_41, 0.01, True)
        x_41 = None
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_42 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_45 = torch.nn.functional.leaky_relu(x_44, 0.01, True)
        x_44 = None
        x_46 = x_45 + x_39
        x_45 = x_39 = None
        x_47 = torch.conv2d(
            x_46,
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
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_49 = torch.nn.functional.leaky_relu(x_48, 0.01, True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_52 = torch.nn.functional.leaky_relu(x_51, 0.01, True)
        x_51 = None
        x_53 = x_52 + x_46
        x_52 = x_46 = None
        x_54 = torch.conv2d(
            x_53,
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
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_56 = torch.nn.functional.leaky_relu(x_55, 0.01, True)
        x_55 = None
        x_57 = torch.conv2d(
            x_56,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_56 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_58 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_57 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_59 = torch.nn.functional.leaky_relu(x_58, 0.01, True)
        x_58 = None
        x_60 = x_59 + x_53
        x_59 = x_53 = None
        x_61 = torch.conv2d(
            x_60,
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
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_63 = torch.nn.functional.leaky_relu(x_62, 0.01, True)
        x_62 = None
        x_64 = torch.conv2d(
            x_63,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_63 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_64 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_66 = torch.nn.functional.leaky_relu(x_65, 0.01, True)
        x_65 = None
        x_67 = x_66 + x_60
        x_66 = x_60 = None
        x_68 = torch.conv2d(
            x_67,
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
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_70 = torch.nn.functional.leaky_relu(x_69, 0.01, True)
        x_69 = None
        x_71 = torch.conv2d(
            x_70,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_70 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_72 = torch.nn.functional.batch_norm(
            x_71,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_71 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_73 = torch.nn.functional.leaky_relu(x_72, 0.01, True)
        x_72 = None
        x_74 = x_73 + x_67
        x_73 = x_67 = None
        x_75 = torch.conv2d(
            x_74,
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
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_77 = torch.nn.functional.leaky_relu(x_76, 0.01, True)
        x_76 = None
        x_78 = torch.conv2d(
            x_77,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_77 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_79 = torch.nn.functional.batch_norm(
            x_78,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_78 = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_80 = torch.nn.functional.leaky_relu(x_79, 0.01, True)
        x_79 = None
        x_81 = x_80 + x_74
        x_80 = x_74 = None
        x_82 = torch.conv2d(
            x_81,
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
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_84 = torch.nn.functional.leaky_relu(x_83, 0.01, True)
        x_83 = None
        x_85 = torch.conv2d(
            x_84,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_84 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_85 = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_87 = torch.nn.functional.leaky_relu(x_86, 0.01, True)
        x_86 = None
        x_88 = x_87 + x_81
        x_87 = x_81 = None
        input_4 = torch._C._nn.avg_pool2d(x_88, 2, 2, 0, False, True, None)
        x_88 = None
        x_89 = torch.conv2d(
            input_4,
            l_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_4 = l_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_conv_parameters_weight_ = (None)
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_89 = l_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_conv_down_modules_1_modules_bn_parameters_bias_ = (None)
        x_91 = torch.nn.functional.leaky_relu(x_90, 0.01, True)
        x_90 = None
        x_92 = torch.conv2d(
            x_91,
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
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_94 = torch.nn.functional.leaky_relu(x_93, 0.01, True)
        x_93 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_94 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_97 = torch.nn.functional.leaky_relu(x_96, 0.01, True)
        x_96 = None
        x_98 = x_97 + x_91
        x_97 = x_91 = None
        x_99 = torch.conv2d(
            x_98,
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
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_99 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_101 = torch.nn.functional.leaky_relu(x_100, 0.01, True)
        x_100 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_101 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_102 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_104 = torch.nn.functional.leaky_relu(x_103, 0.01, True)
        x_103 = None
        x_105 = x_104 + x_98
        x_104 = x_98 = None
        x_106 = torch.conv2d(
            x_105,
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
        x_107 = torch.nn.functional.batch_norm(
            x_106,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_106 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_108 = torch.nn.functional.leaky_relu(x_107, 0.01, True)
        x_107 = None
        x_109 = torch.conv2d(
            x_108,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_108 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_110 = torch.nn.functional.batch_norm(
            x_109,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_109 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_111 = torch.nn.functional.leaky_relu(x_110, 0.01, True)
        x_110 = None
        x_112 = x_111 + x_105
        x_111 = x_105 = None
        x_113 = torch.conv2d(
            x_112,
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
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_115 = torch.nn.functional.leaky_relu(x_114, 0.01, True)
        x_114 = None
        x_116 = torch.conv2d(
            x_115,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_115 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_117 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_116 = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_118 = torch.nn.functional.leaky_relu(x_117, 0.01, True)
        x_117 = None
        x_119 = x_118 + x_112
        x_118 = x_112 = None
        x_120 = torch.conv2d(
            x_119,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_120 = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_122 = torch.nn.functional.leaky_relu(x_121, 0.01, True)
        x_121 = None
        x_123 = torch.conv2d(
            x_122,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_122 = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_4_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_125 = torch.nn.functional.leaky_relu(x_124, 0.01, True)
        x_124 = None
        x_126 = x_125 + x_119
        x_125 = x_119 = None
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_127 = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_129 = torch.nn.functional.leaky_relu(x_128, 0.01, True)
        x_128 = None
        x_130 = torch.conv2d(
            x_129,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_129 = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_130 = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_5_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_132 = torch.nn.functional.leaky_relu(x_131, 0.01, True)
        x_131 = None
        x_133 = x_132 + x_126
        x_132 = x_126 = None
        x_134 = torch.conv2d(
            x_133,
            l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_136 = torch.nn.functional.leaky_relu(x_135, 0.01, True)
        x_135 = None
        x_137 = torch.conv2d(
            x_136,
            l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_136 = l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_137 = l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_6_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_139 = torch.nn.functional.leaky_relu(x_138, 0.01, True)
        x_138 = None
        x_140 = x_139 + x_133
        x_139 = x_133 = None
        x_141 = torch.conv2d(
            x_140,
            l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_141 = l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_143 = torch.nn.functional.leaky_relu(x_142, 0.01, True)
        x_142 = None
        x_144 = torch.conv2d(
            x_143,
            l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_143 = l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_145 = torch.nn.functional.batch_norm(
            x_144,
            l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_144 = l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_7_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_146 = torch.nn.functional.leaky_relu(x_145, 0.01, True)
        x_145 = None
        x_147 = x_146 + x_140
        x_146 = x_140 = None
        input_5 = torch._C._nn.avg_pool2d(x_147, 2, 2, 0, False, True, None)
        x_147 = None
        x_148 = torch.conv2d(
            input_5,
            l_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_5 = l_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_conv_parameters_weight_ = (None)
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = l_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_4_modules_conv_down_modules_1_modules_bn_parameters_bias_ = (None)
        x_150 = torch.nn.functional.leaky_relu(x_149, 0.01, True)
        x_149 = None
        x_151 = torch.conv2d(
            x_150,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_151 = l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_153 = torch.nn.functional.leaky_relu(x_152, 0.01, True)
        x_152 = None
        x_154 = torch.conv2d(
            x_153,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_153 = l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_4_modules_blocks_modules_0_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_156 = torch.nn.functional.leaky_relu(x_155, 0.01, True)
        x_155 = None
        x_157 = x_156 + x_150
        x_156 = x_150 = None
        x_158 = torch.conv2d(
            x_157,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_159 = torch.nn.functional.batch_norm(
            x_158,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_158 = l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_160 = torch.nn.functional.leaky_relu(x_159, 0.01, True)
        x_159 = None
        x_161 = torch.conv2d(
            x_160,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_160 = l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_162 = torch.nn.functional.batch_norm(
            x_161,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_161 = l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_4_modules_blocks_modules_1_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_163 = torch.nn.functional.leaky_relu(x_162, 0.01, True)
        x_162 = None
        x_164 = x_163 + x_157
        x_163 = x_157 = None
        x_165 = torch.conv2d(
            x_164,
            l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_166 = torch.nn.functional.batch_norm(
            x_165,
            l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_165 = l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_167 = torch.nn.functional.leaky_relu(x_166, 0.01, True)
        x_166 = None
        x_168 = torch.conv2d(
            x_167,
            l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_167 = l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_169 = torch.nn.functional.batch_norm(
            x_168,
            l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_168 = l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_4_modules_blocks_modules_2_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_170 = torch.nn.functional.leaky_relu(x_169, 0.01, True)
        x_169 = None
        x_171 = x_170 + x_164
        x_170 = x_164 = None
        x_172 = torch.conv2d(
            x_171,
            l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_173 = torch.nn.functional.batch_norm(
            x_172,
            l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_172 = l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_weight_ = l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv1_modules_bn_parameters_bias_ = (None)
        x_174 = torch.nn.functional.leaky_relu(x_173, 0.01, True)
        x_173 = None
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_174 = l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_176 = torch.nn.functional.batch_norm(
            x_175,
            l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_175 = l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_buffers_running_var_ = l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_weight_ = l_self_modules_stages_modules_4_modules_blocks_modules_3_modules_conv2_modules_bn_parameters_bias_ = (None)
        x_177 = torch.nn.functional.leaky_relu(x_176, 0.01, True)
        x_176 = None
        x_178 = x_177 + x_171
        x_177 = x_171 = None
        x_179 = torch.nn.functional.adaptive_avg_pool2d(x_178, 1)
        x_178 = None
        x_180 = x_179.flatten(1, -1)
        x_179 = None
        x_181 = torch.nn.functional.dropout(x_180, 0.0, False, False)
        x_180 = None
        x_182 = torch._C._nn.linear(
            x_181,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_181 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_182,)
